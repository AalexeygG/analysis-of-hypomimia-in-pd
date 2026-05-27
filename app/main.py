from contextlib import asynccontextmanager
from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal, engine, Base
from app import crud
from app.schemas import SessionOut, MWResult, LOOCVResultOut, LOOCVResponse
from app.processing.pipeline import process_session
from app.processing.ml import run_loocv
from app.auth import (
    TokenOut,
    UserCreate,
    UserOut,
    get_current_user,
    require_admin,
    get_user_by_username,
    create_user,
    verify_password,
    create_access_token,
)
from app.models import User


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def normalize_group(g: str) -> str:
    g = (g or "").strip().upper()
    if g in {"БП", "BP", "PD", "PARKINSON"}:
        return "BP"
    if g in {"КГ", "KG", "CONTROL", "HC"}:
        return "KG"
    return g


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # startup
    Base.metadata.create_all(bind=engine)
    base = Path(settings.DATA_DIR)
    (base / "videos").mkdir(parents=True, exist_ok=True)
    (base / "plots").mkdir(parents=True, exist_ok=True)
    (base / "reports").mkdir(parents=True, exist_ok=True)
    (base / "csv").mkdir(parents=True, exist_ok=True)
    yield
    # shutdown — пока ничего


app = FastAPI(title="BTS Parkinson Mimic Analysis", lifespan=lifespan)

app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)


@app.get("/")
def root():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/auth/register", response_model=UserOut, tags=["auth"])
def register(
    data: UserCreate,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Создание нового пользователя — только администратор."""
    if get_user_by_username(db, data.username):
        raise HTTPException(status_code=400, detail="Пользователь уже существует.")
    return create_user(db, data)


@app.post("/auth/login", response_model=TokenOut, tags=["auth"])
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Вход: возвращает JWT-токен."""
    user = get_user_by_username(db, form.username)
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Неверное имя пользователя или пароль.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Учётная запись деактивирована.")
    token = create_access_token({"sub": user.username})
    return TokenOut(access_token=token)


@app.get("/auth/me", response_model=UserOut, tags=["auth"])
def me(current_user: User = Depends(get_current_user)):
    return current_user


# ---------------------------------------------------------------------------
# Bootstrap: create first admin if no users exist
# ---------------------------------------------------------------------------

@app.post("/auth/bootstrap", response_model=UserOut, tags=["auth"])
def bootstrap(data: UserCreate, db: Session = Depends(get_db)):
    """
    Создаёт первого администратора. Работает только если в БД нет ни одного пользователя.
    После создания первого admin-а этот эндпоинт возвращает 403.
    """
    n_users = db.execute(select(func.count()).select_from(User)).scalar() or 0
    if n_users > 0:
        raise HTTPException(
            status_code=403,
            detail="Первый пользователь уже создан. Используйте /auth/register от имени admin.",
        )
    data.role = "admin"
    return create_user(db, data)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@app.post("/api/sessions", response_model=SessionOut, tags=["sessions"])
def create_and_process_session(
    external_id: str = Form(...),
    group: str = Form(...),
    test_type: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    group = normalize_group(group)
    if group not in {"BP", "KG"}:
        raise HTTPException(status_code=400, detail="Группа должна быть BP или KG.")
    if test_type not in {"smile", "blink"}:
        raise HTTPException(status_code=400, detail="Проба должна быть smile или blink.")

    patient = crud.get_patient_by_external_id(db, external_id)
    if patient is None:
        patient = crud.create_patient(db, external_id, group)
    else:
        if patient.group != group:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Пациент {external_id} уже зарегистрирован в группе {patient.group}. "
                    f"Для другой группы используйте другой ID."
                ),
            )

    data_dir = Path(settings.DATA_DIR)
    videos_dir = data_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    fname = Path(file.filename).name.replace(":", "-")
    video_path = videos_dir / f"patient_{patient.id}_upload_{fname}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    sess = crud.create_session(db, patient.id, test_type, str(video_path))

    try:
        bp_sessions = crud.get_sessions_done_by_test_and_group(db, test_type, "BP")
        kg_sessions = crud.get_sessions_done_by_test_and_group(db, test_type, "KG")

        bp_features = [crud.get_features_for_session(db, s.id) for s in bp_sessions]
        kg_features = [crud.get_features_for_session(db, s.id) for s in kg_sessions]

        plots_dir = data_dir / "plots"
        reports_dir = data_dir / "reports"
        csv_dir = data_dir / "csv"

        res = process_session(
            session_id=sess.id,
            video_path=str(video_path),
            test_type=test_type,
            group=group,
            min_quality=settings.MIN_QUALITY_SCORE,
            bp_sessions_features=bp_features,
            kg_sessions_features=kg_features,
            plots_dir=plots_dir,
            reports_dir=reports_dir,
            csv_dir=csv_dir,
        )

        crud.replace_features(db, sess.id, res.features)
        crud.set_session_done(
            db,
            sess.id,
            res.quality_score,
            res.dynamics_plot_path,
            res.box_plot_path,
            res.report_path,
            res.csv_path,
        )

        s2 = crud.get_session(db, sess.id)
        feats = crud.get_features_for_session(db, sess.id)

        return SessionOut(
            id=s2.id,
            patient_id=s2.patient_id,
            test_type=s2.test_type,
            status=s2.status,
            quality_score=s2.quality_score,
            error_message=s2.error_message,
            features=feats,
            stats={k: MWResult(**v) for k, v in res.stats.items()},
            conclusion=res.conclusion_text,
            risk_score=res.risk_score,
            report_path=s2.report_path,
            csv_path=s2.csv_path,
        )

    except Exception as e:
        crud.set_session_error(db, sess.id, str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Downloads (protected)
# ---------------------------------------------------------------------------

@app.get("/download/report/{session_id}", tags=["downloads"])
def download_report(
    session_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    s = crud.get_session(db, session_id)
    if not s or not s.report_path:
        raise HTTPException(status_code=404, detail="Отчёт не найден.")
    path = Path(s.report_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл отчёта отсутствует на диске.")
    return FileResponse(
        path=path,
        filename=path.name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.get("/download/csv/{session_id}", tags=["downloads"])
def download_csv(
    session_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    s = crud.get_session(db, session_id)
    if not s or not s.csv_path:
        raise HTTPException(status_code=404, detail="CSV не найден.")
    path = Path(s.csv_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл CSV отсутствует на диске.")
    return FileResponse(path=path, filename=path.name, media_type="text/csv")


@app.get("/download/plot/mean/{session_id}", tags=["downloads"])
def download_plot_dynamics(
    session_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    s = crud.get_session(db, session_id)
    if not s or not s.mean_plot_path:
        raise HTTPException(status_code=404, detail="График не найден.")
    path = Path(s.mean_plot_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл графика отсутствует на диске.")
    return FileResponse(path=path, filename=path.name, media_type="image/png")


# ---------------------------------------------------------------------------
# ML / LOOCV
# ---------------------------------------------------------------------------

@app.post("/api/ml/loocv", response_model=LOOCVResponse, tags=["ml"])
def run_ml_loocv(
    test_type: str = Form(...),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """
    Запускает LOOCV (каждое видео — отдельный фолд) на накопленных данных из БД.
    Возвращает Accuracy, ROC-AUC, F1, confusion matrix и важность признаков (RF).
    """
    if test_type not in {"smile", "blink"}:
        raise HTTPException(status_code=400, detail="Проба должна быть smile или blink.")

    bp_sessions = crud.get_sessions_done_by_test_and_group(db, test_type, "BP")
    kg_sessions = crud.get_sessions_done_by_test_and_group(db, test_type, "KG")

    bp_features = [crud.get_features_for_session(db, s.id) for s in bp_sessions]
    kg_features = [crud.get_features_for_session(db, s.id) for s in kg_sessions]

    # Фильтруем пустые записи
    bp_features = [f for f in bp_features if f]
    kg_features = [f for f in kg_features if f]

    try:
        loocv_results = run_loocv(bp_features, kg_features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return LOOCVResponse(
        test_type=test_type,
        results=[
            LOOCVResultOut(
                classifier=r.classifier,
                n_samples=r.n_samples,
                n_bp=r.n_bp,
                n_kg=r.n_kg,
                accuracy=r.accuracy,
                f1=r.f1,
                roc_auc=r.roc_auc if not (r.roc_auc != r.roc_auc) else None,  # NaN → None
                pr_auc=r.pr_auc if not (r.pr_auc != r.pr_auc) else None,  # NaN → None
                confusion=r.confusion,
                feature_importance=r.feature_importance,
            )
            for r in loocv_results
        ],
    )


@app.get("/download/plot/box/{session_id}", tags=["downloads"])
def download_plot_box(
    session_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    s = crud.get_session(db, session_id)
    if not s or not s.box_plot_path:
        raise HTTPException(status_code=404, detail="Boxplot не найден.")
    path = Path(s.box_plot_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл графика отсутствует на диске.")
    return FileResponse(path=path, filename=path.name, media_type="image/png")
