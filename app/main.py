from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import json
from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from .auth import (
    SESSION_COOKIE,
    authenticate_user,
    get_current_user,
    require_role,
    seed_default_users,
)
from .db import init_db, get_db, ChatLog, User, DocRequest

# ---------- App setup ----------
app = FastAPI(title="PFA RAG Assistant (MVP)")

BASE_DIR = Path(__file__).resolve().parent
templates_dir = BASE_DIR / "templates"
static_dir = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(templates_dir))
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    from .db import SessionLocal

    db = SessionLocal()
    try:
        seed_default_users(db)
    finally:
        db.close()


# ---------- Auth ----------
@app.get("/", response_class=HTMLResponse)
def root(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return RedirectResponse(url="/chat", status_code=302)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if user:
        return RedirectResponse(url="/chat", status_code=302)

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "user": None, "error": request.query_params.get("error")},
    )


@app.post("/login")
def login_action(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = authenticate_user(db, username=username.strip(), password=password)
    if not user:
        return RedirectResponse(url="/login?error=1", status_code=302)

    resp = RedirectResponse(url="/chat", status_code=302)
    resp.set_cookie(
        key=SESSION_COOKIE,
        value=user.username,
        httponly=True,
        samesite="lax",
    )
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


# ---------- Chat ----------
@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    logs = (
        db.query(ChatLog)
        .filter(ChatLog.user_id == user.id)
        .order_by(ChatLog.created_at.desc())
        .limit(10)
        .all()
    )

    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "user": user, "logs": list(reversed(logs))},
    )


@app.post("/chat")
def chat_action(
    request: Request,
    question: str = Form(...),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    q = question.strip()

    from .rag import answer_question

    try:
        answer, _ = answer_question(q, top_k=3)
    except Exception as e:
        answer = f"⚠️ Erreur RAG/LLM: {e}"

    db.add(ChatLog(user_id=user.id, question=q, answer=answer))
    db.commit()

    return RedirectResponse(url="/chat", status_code=302)


# ---------- Requests workflow (Student -> Admin -> Teacher) ----------
@app.get("/requests", response_class=HTMLResponse)
def student_requests(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    require_role(user, {"student"})

    reqs = (
        db.query(DocRequest)
        .filter(DocRequest.student_id == user.id)
        .order_by(DocRequest.created_at.desc())
        .all()
    )

    return templates.TemplateResponse(
        "requests_student.html",
        {
            "request": request,
            "user": user,
            "reqs": reqs,
            "today": date.today().strftime("%d/%m/%Y"),
        },
    )


@app.post("/requests/new")
def student_request_new(
    request: Request,
    student_full_name: str = Form(...),
    student_cne: str = Form(...),
    student_cin: str = Form(...),
    student_level: str = Form(...),
    company_name: str = Form(...),
    company_city: str = Form(""),
    internship_topic: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    extra_note: str = Form(""),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    require_role(user, {"student"})

    payload = {
        "student_full_name": student_full_name.strip(),
        "student_cne": student_cne.strip(),
        "student_cin": student_cin.strip(),
        "student_level": student_level.strip(),
        "company_name": company_name.strip(),
        "company_city": (company_city or "").strip(),
        "internship_topic": internship_topic.strip(),
        "start_date": start_date.strip(),
        "end_date": end_date.strip(),
        "extra_note": (extra_note or "").strip(),
    }

    dr = DocRequest(
        doc_type="internship_attestation",
        status="pending",
        student_id=user.id,
        payload_json=json.dumps(payload, ensure_ascii=False),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(dr)
    db.commit()

    return RedirectResponse(url="/requests", status_code=302)


@app.get("/requests/admin", response_class=HTMLResponse)
def admin_requests(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    require_role(user, {"admin"})

    reqs = (
        db.query(DocRequest)
        .filter(DocRequest.doc_type == "internship_attestation")
        .order_by(DocRequest.created_at.desc())
        .all()
    )

    return templates.TemplateResponse(
        "requests_admin.html",
        {"request": request, "user": user, "reqs": reqs},
    )


@app.post("/requests/admin/{req_id}/approve")
def admin_approve(req_id: int, request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    require_role(user, {"admin"})

    dr = db.query(DocRequest).filter(DocRequest.id == req_id).first()
    if not dr:
        return Response(status_code=404)

    dr.status = "approved"
    dr.approved_by_id = user.id
    dr.updated_at = datetime.utcnow()
    db.add(dr)
    db.commit()

    return RedirectResponse(url="/requests/admin", status_code=302)


@app.post("/requests/admin/{req_id}/refuse")
def admin_refuse(
    req_id: int,
    request: Request,
    admin_comment: str = Form(""),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    require_role(user, {"admin"})

    dr = db.query(DocRequest).filter(DocRequest.id == req_id).first()
    if not dr:
        return Response(status_code=404)

    dr.status = "refused"
    dr.admin_comment = (admin_comment or "").strip()
    dr.approved_by_id = user.id
    dr.updated_at = datetime.utcnow()
    db.add(dr)
    db.commit()

    return RedirectResponse(url="/requests/admin", status_code=302)


@app.get("/requests/teacher", response_class=HTMLResponse)
def teacher_requests(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    require_role(user, {"teacher"})

    reqs = (
        db.query(DocRequest)
        .filter(DocRequest.doc_type == "internship_attestation")
        .filter(DocRequest.status.in_(["approved", "generated"]))
        .order_by(DocRequest.updated_at.desc())
        .all()
    )

    return templates.TemplateResponse(
        "requests_teacher.html",
        {"request": request, "user": user, "reqs": reqs},
    )


# ---------- Dashboard ----------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    require_role(user, {"admin"})

    total_logs = db.query(ChatLog).count()

    # Logs par rôle (simple)
    roles_count = {"student": 0, "teacher": 0, "admin": 0}
    users = db.query(User).all()
    for u in users:
        cnt = db.query(ChatLog).filter(ChatLog.user_id == u.id).count()
        roles_count[u.role] = roles_count.get(u.role, 0) + cnt

    # Top 5 questions
    top = (
        db.query(ChatLog.question, func.count(ChatLog.id).label("cnt"))
        .group_by(ChatLog.question)
        .order_by(func.count(ChatLog.id).desc())
        .limit(5)
        .all()
    )
    top_questions = [{"question": q, "count": c} for (q, c) in top]

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "total_logs": total_logs,
            "roles": roles_count,
            "top_questions": top_questions,
        },
    )


# ---------- Document generator (Attestation) ----------
@app.get("/generate", response_class=HTMLResponse)
def generate_page(
    request: Request,
    req_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    require_role(user, {"teacher"})

    today = date.today().strftime("%d/%m/%Y")

    prefill: dict = {}
    request_id: Optional[int] = None
    request_status: Optional[str] = None

    if req_id is not None:
        dr = db.query(DocRequest).filter(DocRequest.id == req_id).first()
        if dr and dr.status in ("approved", "generated"):
            prefill = dr.payload()
            request_id = dr.id
            request_status = dr.status

    return templates.TemplateResponse(
        "generate.html",
        {
            "request": request,
            "user": user,
            "today": today,
            "prefill": prefill,
            "request_id": request_id,
            "request_status": request_status,
        },
    )


def _bool_from_checkbox(value: Optional[str]) -> bool:
    return bool(value)


def _build_attestation_data(
    *,
    school_name: str,
    school_city: str,
    student_full_name: str,
    student_cne: str,
    student_cin: str,
    student_level: str,
    company_name: str,
    internship_topic: str,
    start_date: str,
    end_date: str,
    extra_note: str,
    signer_name: str,
    issue_date: str,
    ai_improve: Optional[str],
):
    from .docgen import AttestationData

    data = AttestationData(
        school_name=school_name.strip(),
        city=school_city.strip(),
        student_full_name=student_full_name.strip(),
        student_cne=student_cne.strip(),
        student_cin=student_cin.strip(),
        student_level=student_level.strip(),
        company_name=company_name.strip(),
        internship_topic=internship_topic.strip(),
        start_date=start_date.strip(),
        end_date=end_date.strip(),
        extra_note=(extra_note or "").strip(),
        signatory_name=signer_name.strip(),
        signature_city=school_city.strip(),
        signature_date=issue_date.strip(),
        ai_improve=_bool_from_checkbox(ai_improve),
    )
    return data


@app.post("/generate/preview", response_class=HTMLResponse)
def generate_preview(
    request: Request,
    request_id: Optional[int] = Form(None),  # ✅ added
    school_name: str = Form(...),
    school_city: str = Form(...),
    student_full_name: str = Form(...),
    student_cne: str = Form(...),
    student_cin: str = Form(...),
    student_level: str = Form(...),
    company_name: str = Form(...),
    company_city: str = Form(""),            # optional for HTML only
    internship_topic: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    extra_note: str = Form(""),
    signer_name: str = Form(...),
    issue_date: str = Form(...),
    ai_improve: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    require_role(user, {"teacher"})

    from .docgen import build_attestation_body

    data = _build_attestation_data(
        school_name=school_name,
        school_city=school_city,
        student_full_name=student_full_name,
        student_cne=student_cne,
        student_cin=student_cin,
        student_level=student_level,
        company_name=company_name,
        internship_topic=internship_topic,
        start_date=start_date,
        end_date=end_date,
        extra_note=extra_note,
        signer_name=signer_name,
        issue_date=issue_date,
        ai_improve=ai_improve,
    )

    d = data.to_dict()

    # IA optionnelle
    if data.ai_improve and d.get("extra_note", "").strip():
        from .docgen import improve_text_with_llm
        d["extra_note"] = improve_text_with_llm(d["extra_note"], timeout_s=180)

    body_text = build_attestation_body(d)

    payload = {
        "request": request,
        "user": user,
        **d,

        # ✅ keep request_id for preview (optional)
        "request_id": request_id,

        # ✅ extra field for HTML
        "company_city": (company_city or "").strip(),

        # aliases for template
        "school_city": d.get("city", ""),
        "signer_name": d.get("signatory_name", ""),
        "issue_date": d.get("signature_date", ""),
        "body_text": body_text,
    }
    return templates.TemplateResponse("attestation.html", payload)


@app.post("/generate/improve")
def generate_improve(
    request: Request,
    extra_note: str = Form(""),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return Response(status_code=401)

    require_role(user, {"teacher"})

    from .docgen import improve_text_with_llm
    improved = improve_text_with_llm(extra_note, timeout_s=180)

    return {"improved": improved}


@app.post("/generate/pdf")
def generate_pdf(
    request: Request,
    request_id: Optional[int] = Form(None),  # ✅ added
    school_name: str = Form(...),
    school_city: str = Form(...),
    student_full_name: str = Form(...),
    student_cne: str = Form(...),
    student_cin: str = Form(...),
    student_level: str = Form(...),
    company_name: str = Form(...),
    internship_topic: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    extra_note: str = Form(""),
    signer_name: str = Form(...),
    issue_date: str = Form(...),
    ai_improve: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    require_role(user, {"teacher"})

    from .docgen import render_attestation_pdf

    data = _build_attestation_data(
        school_name=school_name,
        school_city=school_city,
        student_full_name=student_full_name,
        student_cne=student_cne,
        student_cin=student_cin,
        student_level=student_level,
        company_name=company_name,
        internship_topic=internship_topic,
        start_date=start_date,
        end_date=end_date,
        extra_note=extra_note,
        signer_name=signer_name,
        issue_date=issue_date,
        ai_improve=ai_improve,
    )

    pdf_bytes = render_attestation_pdf(data, improve_note=data.ai_improve)
    filename = f"attestation_{data.student_full_name.replace(' ', '_')}.pdf"

    # ✅ If PDF from an approved request -> mark generated
    if request_id:
        dr = db.query(DocRequest).filter(DocRequest.id == request_id).first()
        if dr and dr.status == "approved":
            dr.status = "generated"
            dr.processed_by_id = user.id
            dr.updated_at = datetime.utcnow()
            db.add(dr)
            db.commit()

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
