from __future__ import annotations

"""Document generation (Étape 4).

- Prévisualisation HTML via template Jinja2
- Export PDF via ReportLab (sans dépendance externe)

Option : amélioration d'un texte libre via llama.cpp server.
"""

from dataclasses import dataclass, asdict
from datetime import date
from io import BytesIO
from typing import Any, Dict, Optional

import os

import requests
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors


LLAMA_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
LLAMA_COMPLETION_ENDPOINT = f"{LLAMA_URL}/completion"


@dataclass
class AttestationData:
    school_name: str = "EMSI Rabat"
    city: str = "Rabat"

    # student
    student_full_name: str = ""
    student_cne: str = ""
    student_cin: str = ""
    student_level: str = ""

    # internship
    company_name: str = ""
    company_city: str = ""
    company_supervisor: str = ""
    internship_topic: str = ""
    start_date: str = ""
    end_date: str = ""

    # signature
    signatory_name: str = ""
    signatory_role: str = "Direction"
    signature_city: str = "Rabat"
    signature_date: str = ""

    # optional paragraph (can be improved by LLM)
    extra_note: str = ""

    # UI flag (checkbox)
    ai_improve: bool = False


    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d.get("signature_date"):
            d["signature_date"] = date.today().isoformat()
        return d


def improve_text_with_llm(text: str, timeout_s: int = 300) -> str:
    """Optionnel : reformule un texte en français administratif.

    Si le serveur LLM est indisponible, on renvoie le texte original.
    """
    if not text.strip():
        return text

    prompt = (
        "Tu es un assistant administratif d'une école d'ingénieurs. "
        "Réécris le texte suivant en français formel, clair, et concis (2 à 4 phrases). "
        "Ne change pas les faits.\n\n"
        f"TEXTE:\n{text.strip()}\n"
    )

    payload = {
        "prompt": prompt,
        "n_predict": 120,
        "temperature": 0.2,
        "stream": False,
        # stop early on common endings
        "stop": ["</s>"]
    }

    try:
        r = requests.post(LLAMA_COMPLETION_ENDPOINT, json=payload, timeout=(10, timeout_s))
        if r.status_code != 200:
            return text
        data = r.json()
        out = (data.get("content") or "").strip()
        return out or text
    except Exception:
        return text


def build_attestation_body(data: Dict[str, Any]) -> str:
    """Texte principal de l'attestation (standard)."""
    student = data.get("student_full_name", "").strip() or "[Nom & Prénom]"
    cne = data.get("student_cne", "").strip()
    cin = data.get("student_cin", "").strip()
    level = data.get("student_level", "").strip()

    company = data.get("company_name", "").strip() or "[Organisme d'accueil]"
    topic = data.get("internship_topic", "").strip()
    start = data.get("start_date", "").strip()
    end = data.get("end_date", "").strip()

    id_part = []
    if cne:
        id_part.append(f"CNE : {cne}")
    if cin:
        id_part.append(f"CIN : {cin}")
    id_part = " — ".join(id_part)

    level_part = f" en {level}" if level else ""
    topic_part = f" sur le thème : \"{topic}\"" if topic else ""

    period_part = ""
    if start and end:
        period_part = f"durant la période allant du {start} au {end}"
    elif start:
        period_part = f"à partir du {start}"
    elif end:
        period_part = f"jusqu'au {end}"

    parts = [
        f"Nous attestons que l'étudiant(e) {student}{level_part}",
    ]
    if id_part:
        parts.append(f"({id_part})")

    parts.append(f"effectue un stage au sein de {company}{topic_part}.")

    if period_part:
        parts.append(period_part + ".")

    return " ".join(parts)


def render_attestation_pdf(data: AttestationData, improve_note: bool = False) -> bytes:
    """Generate a cleaner, more professional PDF attestation (ReportLab only)."""
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT

    d = data.to_dict()

    if improve_note and d.get("extra_note", "").strip():
        d["extra_note"] = improve_text_with_llm(d["extra_note"], timeout_s=600)

    school_name = (d.get("school_name") or "EMSI").strip()
    title = "ATTESTATION DE STAGE"

    # Data
    student = (d.get("student_full_name") or "").strip()
    level = (d.get("student_level") or "").strip()
    cne = (d.get("student_cne") or "").strip()
    cin = (d.get("student_cin") or "").strip()

    company = (d.get("company_name") or "").strip()
    topic = (d.get("internship_topic") or "").strip()
    start = (d.get("start_date") or "").strip()
    end = (d.get("end_date") or "").strip()

    extra = (d.get("extra_note") or "").strip()

    sign_city = (d.get("signature_city") or d.get("city") or "").strip()
    sign_date = (d.get("signature_date") or "").strip()
    signatory_name = (d.get("signatory_name") or "").strip()
    signatory_role = (d.get("signatory_role") or "Direction").strip()

    # ---- Canvas header/footer ----
    def draw_decor(canvas, doc):
        w, h = A4
        canvas.saveState()

        # Top band (soft blue)
        canvas.setFillColor(colors.HexColor("#2563eb"))
        canvas.rect(0, h - 1.15 * cm, w, 1.15 * cm, stroke=0, fill=1)

        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(doc.leftMargin, h - 0.78 * cm, school_name)

        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(w - doc.rightMargin, h - 0.78 * cm, "Document administratif — École d’ingénieurs")

        # Watermark (optional, very light)
        try:
            canvas.setFillAlpha(0.06)
        except Exception:
            pass
        canvas.setFillColor(colors.grey)
        canvas.setFont("Helvetica-Bold", 70)
        canvas.saveState()
        canvas.translate(w * 0.18, h * 0.35)
        canvas.rotate(35)
        canvas.drawString(0, 0, "EMSI")
        canvas.restoreState()
        try:
            canvas.setFillAlpha(1)
        except Exception:
            pass

        # Footer
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(doc.leftMargin, 1.1 * cm, "Généré par AI RAG Assistant")
        canvas.drawRightString(w - doc.rightMargin, 1.1 * cm, f"Page {canvas.getPageNumber()}")

        canvas.restoreState()

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=2.2 * cm,     # room for header band
        bottomMargin=2.0 * cm,
        title=title,
        author=school_name,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyPro",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=11.5,
            leading=16,
            alignment=TA_JUSTIFY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MutedSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#6b7280"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="RightSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            alignment=TA_RIGHT,
        )
    )

    story = []

    # Title
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph(title, styles["TitleCenter"]))
    story.append(Spacer(1, 0.2 * cm))

    # Subtitle / metadata
    if sign_city and sign_date:
        story.append(Paragraph(f"Fait à <b>{sign_city}</b>, le <b>{sign_date}</b>", styles["RightSmall"]))
    story.append(Spacer(1, 0.4 * cm))

    # Intro paragraph
    intro = (
        f"Nous soussignés, <b>{school_name}</b>, attestons par la présente que l’étudiant(e) "
        f"<b>{student or '[Nom & Prénom]'}</b>"
        + (f", inscrit(e) en <b>{level}</b>" if level else "")
        + ", a effectué un stage au sein de l’organisme d’accueil mentionné ci-dessous."
    )
    story.append(Paragraph(intro, styles["BodyPro"]))
    story.append(Spacer(1, 0.35 * cm))

    # Info table (labels / values)
    period = ""
    if start and end:
        period = f"Du {start} au {end}"
    elif start:
        period = f"À partir du {start}"
    elif end:
        period = f"Jusqu’au {end}"

    rows = [
        ["Étudiant", student or "—"],
        ["Niveau", level or "—"],
        ["CNE", cne or "—"],
        ["CIN", cin or "—"],
        ["Entreprise d’accueil", company or "—"],
        ["Thème / Sujet", topic or "—"],
        ["Période", period or "—"],
    ]

    table = Table(rows, colWidths=[4.0 * cm, 12.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f4f6")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.45 * cm))

    # Standard body (your existing builder)
    body = build_attestation_body(d)
    story.append(Paragraph(body, styles["BodyPro"]))
    story.append(Spacer(1, 0.35 * cm))

    # Extra note
    if extra:
        story.append(Paragraph("<b>Note :</b> " + extra, styles["BodyPro"]))
        story.append(Spacer(1, 0.35 * cm))

    # Signature block (right)
    sig_lines = []
    if sign_city and sign_date:
        sig_lines.append(f"Fait à <b>{sign_city}</b>, le <b>{sign_date}</b>")
    sig_lines.append(signatory_role or "Direction")
    if signatory_name:
        sig_lines.append(f"<b>{signatory_name}</b>")
    sig_html = "<br/>".join(sig_lines)

    sig_table = Table(
        [
            ["", Paragraph(sig_html, styles["BodyPro"])],
            ["", Paragraph("<font color='#6b7280'>Signature & cachet</font><br/><br/>_________________________", styles["BodyPro"])],
        ],
        colWidths=[10.0 * cm, 6.5 * cm],
    )
    sig_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEABOVE", (1, 0), (1, 0), 0.6, colors.HexColor("#e5e7eb")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(Spacer(1, 0.6 * cm))
    story.append(sig_table)

    doc.build(story, onFirstPage=draw_decor, onLaterPages=draw_decor)
    return buf.getvalue()



def default_attestation_data() -> AttestationData:
    return AttestationData(signature_date=date.today().isoformat())
