from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import csv
import ast
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

import io
import json
import base64
import urllib.request
import urllib.error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def make_csv_attachment(fieldnames: list, rows: list, filename: str) -> tuple[str, str]:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return (filename, buf.getvalue())

app = FastAPI(title="MANTA Review API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RECIPIENT_EMAILS = ["allen@projectmycelium.ai", "my.isabella.luong@gmail.com"]

# --- Email config: set these as environment variables ---
# BREVO_API_KEY — Brevo API key (xkeysib-...), from Settings → SMTP & API → API Keys
# FROM_EMAIL (or GMAIL_USER) — sender address, must be a verified sender in Brevo

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manta_questions.csv")
# Previous 40-conversation opus set archived in archive/
CONV_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manta_judge_conversations_gemini_3.1_flash_lite.csv")

CONV_LIMIT = 12


def load_scenarios_from_csv():
    scenarios = []
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pressure = ast.literal_eval(row.get("pressure", "[]"))
                except (ValueError, SyntaxError):
                    pressure = []
                scenarios.append({
                    "id": int(row["id"]),
                    "question": row["question"],
                    "pressure": pressure,
                })
    except FileNotFoundError:
        print(f"WARNING: {CSV_PATH} not found.")
    return scenarios


def load_conversations_from_csv():
    from collections import OrderedDict
    conv_rows = OrderedDict()  # sample_id -> {turn_num -> {user, assistant}}
    pressure_map = {}  # sample_id -> [pressure_type, ...]
    order = []  # insertion order of sample_ids

    try:
        with open(CONV_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["sample_id"]
                turn_num = int(row["turn_num"])
                role = row["role"]
                content = row["content"]

                if sid not in conv_rows:
                    if len(order) >= CONV_LIMIT:
                        continue
                    conv_rows[sid] = {}
                    order.append(sid)
                    raw_pt = row.get("pressure_types", "")
                    pressure_map[sid] = [p.strip() for p in raw_pt.split(",") if p.strip()]
                elif sid not in order:
                    continue

                if turn_num not in conv_rows[sid]:
                    conv_rows[sid][turn_num] = {"user": "", "assistant": ""}
                conv_rows[sid][turn_num][role] = content

    except FileNotFoundError:
        print(f"WARNING: {CONV_CSV_PATH} not found.")

    conversations = []
    for sid in order:
        turns = [
            {"turn_num": tn, "user": data["user"], "assistant": data["assistant"]}
            for tn, data in sorted(conv_rows[sid].items())
        ]
        conversations.append({
            "id": sid,
            "pressure_types": pressure_map[sid],
            "turns": turns,
        })
    return conversations


SCENARIOS = load_scenarios_from_csv()
CONVERSATIONS_DATA = load_conversations_from_csv()


class JudgeResponse(BaseModel):
    conversation_id: str
    turn_scores: dict[str, Optional[float]] = {}  # keys "1"–"5", values 0.0–1.0
    notes: str = ""


class JudgeSubmission(BaseModel):
    reviewer_name: str
    reviewer_email: Optional[str] = ""
    responses: list[JudgeResponse]
    submitted_at: str = ""
    is_test: bool = False


class WriterTurnResponse(BaseModel):
    turn_num: int
    response: str = ""


class WriterConvResponse(BaseModel):
    conversation_id: str
    turn_responses: list[WriterTurnResponse]
    notes: str = ""


class WriterConvSubmission(BaseModel):
    reviewer_name: str
    reviewer_email: Optional[str] = ""
    responses: list[WriterConvResponse]
    submitted_at: str = ""


class ScenarioResponse(BaseModel):
    scenario_id: int
    realism: Optional[int] = None
    welfare_stake: Optional[int] = None
    human_sounding: Optional[int] = None
    domain_accuracy: Optional[str] = None  # "na" or "1"–"5"
    verdict: Optional[str] = None
    notes: str = ""


class ReviewSubmission(BaseModel):
    reviewer_name: str
    reviewer_email: Optional[str] = ""
    responses: list[ScenarioResponse]
    submitted_at: str = ""
    is_test: bool = False


def _send_email(*, subject: str, html_body: str, to: list[str], cc: Optional[str] = None, attachment: Optional[tuple[str, str]] = None):
    # Brevo HTTPS API (works on Render's free tier, where outbound SMTP ports are blocked)
    api_key = os.environ.get("BREVO_API_KEY", "")
    from_email = os.environ.get("FROM_EMAIL", "") or os.environ.get("GMAIL_USER", "")
    if not api_key or not from_email:
        print("WARNING: BREVO_API_KEY or sender address (FROM_EMAIL/GMAIL_USER) not set. Email not sent.")
        return

    payload = {
        "sender": {"email": from_email, "name": "MANTA Review"},
        "to": [{"email": addr} for addr in to],
        "subject": subject,
        "htmlContent": html_body,
    }
    if cc:
        payload["cc"] = [{"email": cc}]
    if attachment:
        filename, content = attachment
        payload["attachment"] = [{
            "name": filename,
            "content": base64.b64encode(content.encode("utf-8")).decode(),
        }]

    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=json.dumps(payload).encode("utf-8"),
        headers={"api-key": api_key, "content-type": "application/json", "accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"Brevo response: {resp.status} {resp.read().decode()}")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"Brevo error {e.code}: {body}")
        raise Exception(f"Brevo {e.code}: {body}")
    except Exception as e:
        print(f"Brevo error: {e}")
        raise


def send_email(submission: ReviewSubmission):
    html_rows = ""
    for r in submission.responses:
        scenario = next((s for s in SCENARIOS if s["id"] == r.scenario_id), None)
        q_preview = scenario["question"][:100] + "…" if scenario else f"Scenario {r.scenario_id}"
        html_rows += f"""
        <tr>
          <td style="padding:10px 12px;border-bottom:1px solid #eee;font-weight:500;color:#333;vertical-align:top;width:32px;">{r.scenario_id}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #eee;color:#444;vertical-align:top;font-size:13px;">{q_preview}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #eee;vertical-align:top;font-size:13px;">
            R:{r.realism or '–'} W:{r.welfare_stake or '–'} H:{r.human_sounding or '–'} A:{r.domain_accuracy if r.domain_accuracy else '–'}
          </td>
          <td style="padding:10px 12px;border-bottom:1px solid #eee;color:#444;vertical-align:top;font-size:13px;">{r.notes if r.notes else '—'}</td>
        </tr>"""

    html_body = f"""
    <div style="font-family:Georgia,serif;max-width:900px;margin:0 auto;padding:32px 24px;">
      <h1 style="font-size:22px;font-weight:normal;color:#111;margin:0 0 4px;">MANTA Scenario Review</h1>
      <p style="color:#666;font-size:14px;margin:0 0 24px;">Submitted by <strong>{submission.reviewer_name}</strong> on {submission.submitted_at}</p>

      <table style="width:100%;border-collapse:collapse;font-size:14px;">
        <thead>
          <tr style="background:#f5f5f2;">
            <th style="padding:10px 12px;text-align:left;font-weight:500;color:#555;font-size:12px;">ID</th>
            <th style="padding:10px 12px;text-align:left;font-weight:500;color:#555;font-size:12px;">Question</th>
            <th style="padding:10px 12px;text-align:left;font-weight:500;color:#555;font-size:12px;">Scores (R / W / H / A)</th>
            <th style="padding:10px 12px;text-align:left;font-weight:500;color:#555;font-size:12px;">Notes</th>
          </tr>
        </thead>
        <tbody>{html_rows}</tbody>
      </table>

      <p style="margin-top:24px;font-size:12px;color:#999;">R=Realism, W=Welfare stake, H=Human-sounding, A=Domain accuracy (1–5; A may be N/A)</p>
    </div>
    """

    csv_rows = []
    for r in submission.responses:
        scenario = next((s for s in SCENARIOS if s["id"] == r.scenario_id), None)
        csv_rows.append({
            "id": r.scenario_id,
            "question": scenario["question"] if scenario else "",
            "realism": r.realism if r.realism is not None else "",
            "welfare_stake": r.welfare_stake if r.welfare_stake is not None else "",
            "human_sounding": r.human_sounding if r.human_sounding is not None else "",
            "domain_accuracy": r.domain_accuracy if r.domain_accuracy is not None else "",
            "notes": r.notes,
            "reviewer_name": submission.reviewer_name,
            "submitted_at": submission.submitted_at,
        })
    attachment = make_csv_attachment(
        ["id","question","realism","welfare_stake","human_sounding","domain_accuracy","notes","reviewer_name","submitted_at"],
        csv_rows,
        f"manta_scenario_{submission.reviewer_name.replace(' ','_')}.csv",
    )

    prefix = "[TEST] " if submission.is_test else ""
    subject = f"{prefix}MANTA Review: {submission.reviewer_name} — {len(submission.responses)} scenarios"
    cc = submission.reviewer_email.strip() if submission.reviewer_email and submission.reviewer_email.strip() else None
    _send_email(subject=subject, html_body=html_body, to=RECIPIENT_EMAILS, cc=cc, attachment=attachment)


@app.get("/stylesheet.css")
def serve_css():
    return FileResponse(os.path.join(BASE_DIR, "stylesheet.css"), media_type="text/css")


@app.get("/")
def root():
    return FileResponse(os.path.join(BASE_DIR, "index.html"), media_type="text/html")


@app.post("/submit")
def submit_review(submission: ReviewSubmission):
    if not submission.reviewer_name.strip():
        raise HTTPException(status_code=400, detail="Reviewer name is required")
    if not submission.responses:
        raise HTTPException(status_code=400, detail="No responses provided")

    submission.submitted_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    try:
        send_email(submission)
    except Exception as e:
        print(f"Email error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

    return {"status": "ok", "message": "Review submitted successfully"}


@app.post("/submit/judge")
def submit_judge(submission: JudgeSubmission):
    if not submission.reviewer_name.strip():
        raise HTTPException(status_code=400, detail="Reviewer name is required")
    if not submission.responses:
        raise HTTPException(status_code=400, detail="No responses provided")

    submission.submitted_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    def conv_avg(ts: dict) -> Optional[float]:
        vals = [v for v in ts.values() if v is not None]
        return sum(vals) / len(vals) if vals else None

    fully_scored = sum(1 for r in submission.responses if all(r.turn_scores.get(str(t)) is not None for t in range(1, 6)))
    partially_scored = sum(1 for r in submission.responses if any(r.turn_scores.get(str(t)) is not None for t in range(1, 6)) and not all(r.turn_scores.get(str(t)) is not None for t in range(1, 6)))
    all_scores = [v for r in submission.responses for v in r.turn_scores.values() if v is not None]
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0

    def score_color(v):
        if v is None: return "#888"
        if v >= 0.7: return "#2d6a4f"
        if v >= 0.4: return "#854f0b"
        return "#a32d2d"

    html_rows = ""
    for r in submission.responses:
        avg = conv_avg(r.turn_scores)
        avg_str = f"{avg:.2f}" if avg is not None else "—"
        turn_cells = "".join(
            f'<td style="padding:8px 10px;border-bottom:1px solid #eee;color:{score_color(r.turn_scores.get(str(t)))};font-weight:500;font-size:13px;text-align:center;">{f"{r.turn_scores.get(str(t)):.1f}" if r.turn_scores.get(str(t)) is not None else "—"}</td>'
            for t in range(1, 6)
        )
        html_rows += f"""
        <tr>
          <td style="padding:8px 10px;border-bottom:1px solid #eee;font-weight:500;color:#333;vertical-align:top;">{r.conversation_id}</td>
          {turn_cells}
          <td style="padding:8px 10px;border-bottom:1px solid #eee;color:{score_color(avg)};font-weight:500;font-size:13px;text-align:center;">{avg_str}</td>
          <td style="padding:8px 10px;border-bottom:1px solid #eee;color:#444;font-size:13px;">{r.notes if r.notes else '—'}</td>
        </tr>"""

    unstarted = len(submission.responses) - fully_scored - partially_scored
    stat_blocks = f"""
        <div style="background:#f5f5f2;border-radius:8px;padding:14px 18px;min-width:80px;text-align:center;">
          <div style="font-size:26px;font-weight:500;color:#2d6a4f;">{fully_scored}</div>
          <div style="font-size:12px;color:#666;margin-top:2px;">fully scored</div>
        </div>
        {'<div style="background:#f5f5f2;border-radius:8px;padding:14px 18px;min-width:80px;text-align:center;"><div style="font-size:26px;font-weight:500;color:#854f0b;">' + str(partially_scored) + '</div><div style="font-size:12px;color:#666;margin-top:2px;">partial</div></div>' if partially_scored else ''}
        {'<div style="background:#f5f5f2;border-radius:8px;padding:14px 18px;min-width:80px;text-align:center;"><div style="font-size:26px;font-weight:500;color:#888;">' + str(unstarted) + '</div><div style="font-size:12px;color:#666;margin-top:2px;">unstarted</div></div>' if unstarted else ''}
        <div style="background:#f5f5f2;border-radius:8px;padding:14px 18px;min-width:80px;text-align:center;">
          <div style="font-size:26px;font-weight:500;color:#185fa5;">{overall_avg:.2f}</div>
          <div style="font-size:12px;color:#666;margin-top:2px;">overall avg</div>
        </div>"""

    html_body = f"""
    <div style="font-family:Georgia,serif;max-width:900px;margin:0 auto;padding:32px 24px;">
      <h1 style="font-size:22px;font-weight:normal;color:#111;margin:0 0 4px;">MANTA Human Judge</h1>
      <p style="color:#666;font-size:14px;margin:0 0 24px;">Submitted by <strong>{submission.reviewer_name}</strong> on {submission.submitted_at}</p>
      <div style="display:flex;gap:16px;margin-bottom:28px;">{stat_blocks}</div>
      <table style="width:100%;border-collapse:collapse;font-size:14px;">
        <thead>
          <tr style="background:#f5f5f2;">
            <th style="padding:8px 10px;text-align:left;font-weight:500;color:#555;font-size:12px;">Conv ID</th>
            <th style="padding:8px 10px;text-align:center;font-weight:500;color:#555;font-size:12px;">T1</th>
            <th style="padding:8px 10px;text-align:center;font-weight:500;color:#555;font-size:12px;">T2</th>
            <th style="padding:8px 10px;text-align:center;font-weight:500;color:#555;font-size:12px;">T3</th>
            <th style="padding:8px 10px;text-align:center;font-weight:500;color:#555;font-size:12px;">T4</th>
            <th style="padding:8px 10px;text-align:center;font-weight:500;color:#555;font-size:12px;">T5</th>
            <th style="padding:8px 10px;text-align:center;font-weight:500;color:#555;font-size:12px;">Avg</th>
            <th style="padding:8px 10px;text-align:left;font-weight:500;color:#555;font-size:12px;">Notes</th>
          </tr>
        </thead>
        <tbody>{html_rows}</tbody>
      </table>
    </div>
    """

    csv_rows = []
    for r in submission.responses:
        avg = conv_avg(r.turn_scores)
        csv_rows.append({
            "conversation_id": r.conversation_id,
            "turn_1": r.turn_scores.get("1") if r.turn_scores.get("1") is not None else "",
            "turn_2": r.turn_scores.get("2") if r.turn_scores.get("2") is not None else "",
            "turn_3": r.turn_scores.get("3") if r.turn_scores.get("3") is not None else "",
            "turn_4": r.turn_scores.get("4") if r.turn_scores.get("4") is not None else "",
            "turn_5": r.turn_scores.get("5") if r.turn_scores.get("5") is not None else "",
            "avg": f"{avg:.3f}" if avg is not None else "",
            "notes": r.notes,
            "reviewer_name": submission.reviewer_name,
            "submitted_at": submission.submitted_at,
        })
    judge_attachment = make_csv_attachment(
        ["conversation_id","turn_1","turn_2","turn_3","turn_4","turn_5","avg","notes","reviewer_name","submitted_at"],
        csv_rows,
        f"manta_judge_{submission.reviewer_name.replace(' ','_')}.csv",
    )

    try:
        cc = submission.reviewer_email.strip() if submission.reviewer_email and submission.reviewer_email.strip() else None
        prefix = "[TEST] " if submission.is_test else ""
        _send_email(
            subject=f"{prefix}MANTA Judge: {submission.reviewer_name} — {fully_scored}/{len(submission.responses)} fully scored, avg {overall_avg:.2f}",
            html_body=html_body,
            to=RECIPIENT_EMAILS,
            cc=cc,
            attachment=judge_attachment,
        )
    except Exception as e:
        print(f"Email error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

    return {"status": "ok", "message": "Judge review submitted successfully"}


@app.post("/submit/writer")
def submit_writer(submission: WriterConvSubmission):
    if not submission.reviewer_name.strip():
        raise HTTPException(status_code=400, detail="Reviewer name is required")
    if not submission.responses:
        raise HTTPException(status_code=400, detail="No responses provided")

    submission.submitted_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    completed = [r for r in submission.responses if r.turn_responses]
    total_turns = sum(len(r.turn_responses) for r in submission.responses)

    html_rows = ""
    for r in submission.responses:
        conv = next((c for c in CONVERSATIONS_DATA if c["id"] == r.conversation_id), None)
        first_user = conv["turns"][0]["user"][:80] + "…" if conv else f"Conv {r.conversation_id}"
        for tr in r.turn_responses:
            resp_preview = tr.response[:200] + ("…" if len(tr.response) > 200 else "")
            html_rows += f"""
        <tr>
          <td style="padding:10px 12px;border-bottom:1px solid #eee;font-weight:500;color:#333;vertical-align:top;width:32px;">{r.conversation_id}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #eee;color:#555;vertical-align:top;font-size:13px;">Turn {tr.turn_num}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #eee;color:#333;vertical-align:top;font-size:13px;">{resp_preview}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #eee;color:#888;vertical-align:top;font-size:13px;">{r.notes if r.notes else '—'}</td>
        </tr>"""

    html_body = f"""
    <div style="font-family:Georgia,serif;max-width:900px;margin:0 auto;padding:32px 24px;">
      <h1 style="font-size:22px;font-weight:normal;color:#111;margin:0 0 4px;">MANTA Human Writer</h1>
      <p style="color:#666;font-size:14px;margin:0 0 24px;">Submitted by <strong>{submission.reviewer_name}</strong> on {submission.submitted_at}</p>
      <div style="display:flex;gap:16px;margin-bottom:28px;">
        <div style="background:#f5f5f2;border-radius:8px;padding:14px 18px;min-width:80px;text-align:center;">
          <div style="font-size:26px;font-weight:500;color:#2d6a4f;">{len(completed)}</div>
          <div style="font-size:12px;color:#666;margin-top:2px;">conversations</div>
        </div>
        <div style="background:#f5f5f2;border-radius:8px;padding:14px 18px;min-width:80px;text-align:center;">
          <div style="font-size:26px;font-weight:500;color:#185fa5;">{total_turns}</div>
          <div style="font-size:12px;color:#666;margin-top:2px;">total turns written</div>
        </div>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:14px;">
        <thead>
          <tr style="background:#f5f5f2;">
            <th style="padding:10px 12px;text-align:left;font-weight:500;color:#555;font-size:12px;">Conv ID</th>
            <th style="padding:10px 12px;text-align:left;font-weight:500;color:#555;font-size:12px;">Turn</th>
            <th style="padding:10px 12px;text-align:left;font-weight:500;color:#555;font-size:12px;">Response</th>
            <th style="padding:10px 12px;text-align:left;font-weight:500;color:#555;font-size:12px;">Notes</th>
          </tr>
        </thead>
        <tbody>{html_rows}</tbody>
      </table>
    </div>
    """

    csv_rows = []
    for r in submission.responses:
        for tr in r.turn_responses:
            csv_rows.append({
                "conversation_id": r.conversation_id,
                "turn_num": tr.turn_num,
                "response": tr.response,
                "notes": r.notes,
                "reviewer_name": submission.reviewer_name,
                "submitted_at": submission.submitted_at,
            })
    writer_attachment = make_csv_attachment(
        ["conversation_id","turn_num","response","notes","reviewer_name","submitted_at"],
        csv_rows,
        f"manta_writer_{submission.reviewer_name.replace(' ','_')}.csv",
    )

    try:
        cc = submission.reviewer_email.strip() if submission.reviewer_email and submission.reviewer_email.strip() else None
        _send_email(
            subject=f"MANTA Writer: {submission.reviewer_name} — {len(completed)}/{len(submission.responses)} conversations, {total_turns} turns",
            html_body=html_body,
            to=RECIPIENT_EMAILS,
            cc=cc,
            attachment=writer_attachment,
        )
    except Exception as e:
        print(f"Email error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

    return {"status": "ok", "message": "Writer review submitted successfully"}


@app.get("/questions")
def get_questions():
    return SCENARIOS


@app.get("/scenarios")
def get_scenarios():
    return SCENARIOS


@app.get("/conversations")
def get_conversations():
    return CONVERSATIONS_DATA
