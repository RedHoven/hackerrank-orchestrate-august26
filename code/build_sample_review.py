import csv
import html
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
ACTIONS = ("notify", "digest", "mute")
MESSAGE_TYPES = ("personal", "urgent", "event", "payment", "business_update", "promotion", "greeting", "forward", "spam", "scam", "unknown")


def rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def index(items, key):
    return {item[key]: item for item in items}


def table(item):
    return "<dl>" + "".join(
        f"<dt>{html.escape(key.replace('_', ' '))}</dt><dd>{html.escape(value or '—')}</dd>"
        for key, value in item.items()
    ) + "</dl>"


def prediction(title, item, gt):
    action = item["predicted_action"]
    kind = item["predicted_message_type"]
    match = "match" if action == gt["action"] and kind == gt["message_type"] else "miss"
    return f'''<section class="prediction {match}"><h3>{title}</h3>
      <p><span class="badge {action}">{html.escape(action)}</span> <span class="type">{html.escape(kind)}</span>
      <span class="confidence">{html.escape(item["predicted_confidence"])}</span></p>
      <p>{html.escape(item["predicted_reason"])}</p>
      <p class="evidence">Evidence: {html.escape(item["predicted_evidence_message_ids"] or "none")}</p></section>'''


def target_prediction(item, review):
    reviewed_action = review.get("reviewed_action") or item["action"]
    reviewed_type = review.get("reviewed_message_type") or item["message_type"]
    action_options = "".join(f'<option value="{action}"{" selected" if action == reviewed_action else ""}>{action}</option>' for action in ACTIONS)
    type_options = "".join(f'<option value="{kind}"{" selected" if kind == reviewed_type else ""}>{kind}</option>' for kind in MESSAGE_TYPES)
    verdict = review.get("review_verdict", "not_reviewed")
    review_html = f'''<section class="review {html.escape(verdict)}"><h3>Codex review</h3>
      <p><strong>{html.escape(verdict.replace('_', ' '))}</strong> · Recommended: <span class="badge {html.escape(reviewed_action)}">{html.escape(reviewed_action)}</span> <span class="type">{html.escape(reviewed_type)}</span></p>
      <p><b>Action:</b> {html.escape(review.get("action_review") or "Not reviewed")}</p>
      <p><b>Type:</b> {html.escape(review.get("type_review") or "Not reviewed")}</p>
      <p><b>Evidence:</b> {html.escape(review.get("evidence_review") or "Not reviewed")}</p>
      <p><b>Confidence:</b> {html.escape(review.get("confidence_review") or "Not reviewed")}</p>
      {f'<p><b>Improvement tags:</b> {html.escape(review["improvement_tags"])}</p>' if review.get("improvement_tags") else ''}
      <details><summary>Exact Router 3 input</summary><h3>Complete message</h3><pre>{html.escape(review.get("complete_message") or "—")}</pre><h3>Message context</h3><pre>{html.escape(review.get("router_3_message_context") or "—")}</pre><h3>Nearest labeled examples</h3><pre>{html.escape(review.get("router_3_nearest_labeled_examples") or "—")}</pre><h3>Full prompt</h3><pre>{html.escape(review.get("router_3_prompt") or "—")}</pre></details>
      </section>'''
    return f'''<section class="prediction"><h3>Router 3</h3>
      <p><span class="badge {html.escape(item["action"])}">{html.escape(item["action"])}</span> <span class="type">{html.escape(item["message_type"])}</span>
      <span class="confidence">Action confidence: {html.escape(item["confidence"])}</span></p>
      <p>{html.escape(item["reason"])}</p>
      <p class="evidence">Evidence: {html.escape(item["evidence_message_ids"] or "none")}</p>
      <div class="correction" data-message-id="{html.escape(item["message_id"])}" data-original-action="{html.escape(item["action"])}" data-original-type="{html.escape(item["message_type"])}">
        <label>Correct action <select data-field="action">{action_options}</select></label>
        <label>Correct type <select data-field="message_type">{type_options}</select></label>
        <small data-status>Original prediction</small>
      </div></section>{review_html}'''


def review_metrics(target, reviews):
    labeled = [
        (prediction, reviews.get(message_id, {}))
        for message_id, prediction in target.items()
        if reviews.get(message_id, {}).get("reviewed_action") and reviews.get(message_id, {}).get("reviewed_message_type")
    ]
    if not labeled:
        return "<p class=\"subtitle\">No Codex-reviewed labels are available yet.</p>"
    total = len(labeled)
    action = sum(prediction["action"] == review["reviewed_action"] for prediction, review in labeled)
    message_type = sum(prediction["message_type"] == review["reviewed_message_type"] for prediction, review in labeled)
    combined = sum(prediction["action"] == review["reviewed_action"] and prediction["message_type"] == review["reviewed_message_type"] for prediction, review in labeled)
    metric = lambda label, value: f'<section><strong>{label}</strong><span>{value / total:.1%}</span></section>'
    return f'<section class="metrics"><p>Agreement with Codex review labels · {total} messages</p>{metric("Action", action)}{metric("Message type", message_type)}{metric("Combined", combined)}</section>'


def main():
    samples = rows(DATA / "sample_messages.csv")[:30]
    router0 = index(rows(ROOT / "sample_predictions_baseline.csv"), "message_id")
    router1 = index(rows(ROOT / "sample_predictions_llm_msg.csv"), "message_id")
    router2 = index(rows(ROOT / "sample_predictions_router_2.csv"), "message_id")
    router3 = index(rows(ROOT / "sample_predictions_router_3.csv"), "message_id")
    target = index(rows(ROOT / "output.csv"), "message_id")
    reviews = index(rows(ROOT / "review.csv"), "message_id") if (ROOT / "review.csv").exists() else {}
    metrics = review_metrics(target, reviews)
    users = index(rows(DATA / "users.csv"), "user_id")
    groups = index(rows(DATA / "groups.csv"), "group_id")
    businesses = index(rows(DATA / "business_accounts.csv"), "business_id")
    members = index(rows(DATA / "group_members.csv"), "group_id,user_id") if False else {
        (x["group_id"], x["user_id"]): x for x in rows(DATA / "group_members.csv")}
    images = index(rows(DATA / "image_descriptions.csv"), "image_id")
    voices = index(rows(DATA / "voice_transcriptions.csv"), "voice_note_id")
    events = index(rows(DATA / "message_events.csv"), "message_id")
    history = rows(DATA / "message_history.csv")
    summaries = defaultdict(list)
    for entry in rows(DATA / "daily_notification_summary.csv"):
        summaries[entry["user_id"]].append(entry)

    cards = []
    for n, message in enumerate(samples, 1):
        user = users.get(message["user_id"], {})
        if message["conversation_type"] == "group":
            conversation = groups.get(message["group_id"], {})
            membership = members.get((message["group_id"], message["user_id"]), {})
        else:
            conversation = businesses.get(message["business_id"], {})
            membership = {}
        relevant = [x for x in history if x["user_id"] == message["user_id"] and (
            (message["group_id"] and x["group_id"] == message["group_id"]) or
            (message["business_id"] and x["business_id"] == message["business_id"]) or
            (message["conversation_type"] == "personal" and x["sender_user_id"] == message["sender_user_id"])
        )]
        media = images.get(message["media_id"], {}) if message["media_type"] == "image" else voices.get(message["media_id"], {})
        media_html = ""
        if media:
            label = "Image description" if message["media_type"] == "image" else "Voice transcription"
            media_html = f"<section><h3>{label}</h3><p>{html.escape(media.get('description') or media.get('transcription') or '—')}</p></section>"
        history_html = "<p>None found.</p>" if not relevant else "<ul>" + "".join(
            f"<li><b>{html.escape(x['created_at'])}</b> — {html.escape(x['message_text'] or '[media]')}"
            f"<small>{html.escape(str(events.get(x['message_id'], {}))) if x['message_id'] in events else ''}</small></li>" for x in relevant) + "</ul>"
        activity = summaries[message["user_id"]]
        activity_html = "<p>None found.</p>" if not activity else table({x["date"]: f"sent {x['notifications_sent']}, dismissed {x['notifications_dismissed']}" for x in activity})
        gt = f'''<section class="ground"><h3>Ground truth</h3><p><span class="badge {message['action']}">{html.escape(message['action'])}</span> <span class="type">{html.escape(message['message_type'])}</span> <span class="confidence">{html.escape(message['confidence'])}</span></p><p>{html.escape(message['reason'])}</p><p class="evidence">Evidence: {html.escape(message['evidence_message_ids'])}</p></section>'''
        cards.append(f'''<article><h2>{n:02d}. {html.escape(message['message_id'])}</h2>
          <section><h3>Incoming message</h3><p class="meta">{html.escape(message['created_at'])} · {html.escape(message['conversation_type'])} · sender {html.escape(message['sender_user_id'] or message['business_id'])}</p><p class="message">{html.escape(message['message_text'] or '[No text]')}</p>{media_html}</section>
          <details><summary>Existing context</summary><h3>User</h3>{table(user)}<h3>Sender</h3>{table(users.get(message['sender_user_id'], {})) if message['sender_user_id'] else '<p>Business sender shown below.</p>'}<h3>Conversation</h3>{table(conversation)}{'<h3>Membership</h3>' + table(membership) if membership else ''}<h3>Relevant message history ({len(relevant)})</h3>{history_html}<h3>Notification activity ({len(activity)} days)</h3>{activity_html}<h3>Historical interaction for GT evidence</h3>{table(events.get(message['evidence_message_ids'], {})) if message['evidence_message_ids'] in events else '<p>None found.</p>'}</details>
          <div class="decisions">{gt}{prediction('Router 0', router0[message['message_id']], message)}{prediction('Router 1', router1[message['message_id']], message)}{prediction('Router 2', router2[message['message_id']], message)}{prediction('Router 3', router3[message['message_id']], message)}</div></article>''')
    target_cards = []
    for n, message in enumerate(rows(DATA / "messages.csv"), 1):
        prediction_row = target.get(message["message_id"])
        if not prediction_row:
            continue
        review = reviews.get(message["message_id"], {})
        target_cards.append(f'''<article><h2>{n:03d}. {html.escape(message["message_id"])}</h2>
          <section><h3>Incoming message</h3><p class="meta">{html.escape(message["created_at"])} · {html.escape(message["conversation_type"])} · sender {html.escape(message["sender_user_id"] or message["business_id"])}</p><p class="message">{html.escape(message["message_text"] or "[No text]")}</p></section>
          <div class="decisions target-decisions">{target_prediction(prediction_row, review)}</div></article>''')
    output = ROOT / "sample_router_review.html"
    output.write_text(f'''<!doctype html><html><head><meta charset="utf-8"><title>Router sample review</title><style>
      *{{box-sizing:border-box}} body{{margin:0;background:#f4f6fa;color:#172033;font:15px/1.45 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:28px 18px 60px}}h1{{margin:0}}.subtitle{{color:#56627a}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:16px 0 20px}}.metrics p{{grid-column:1/-1;color:#56627a;margin:0}}.metrics section{{background:#fff;border:1px solid #dce2ec;border-radius:10px;padding:12px}}.metrics strong,.metrics span{{display:block}}.metrics span{{font-size:26px;font-weight:800}}.mode-control{{position:sticky;top:0;z-index:2;background:#f4f6fa;padding:12px 0}}.mode-control label{{display:flex;max-width:440px;align-items:center;gap:10px;font-weight:700}}.mode-control input{{flex:1}}article{{background:#fff;border:1px solid #dce2ec;border-radius:12px;margin:18px 0;padding:20px;box-shadow:0 2px 8px #15203b0a}}h2{{font-size:18px;margin:0 0 14px}}h3{{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#526079;margin:15px 0 6px}}p{{margin:6px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;border:1px solid #dce2ec;border-radius:6px;padding:10px;max-height:420px;overflow:auto}}.meta,.evidence{{font-size:13px;color:#58657a}}.message{{white-space:pre-wrap;background:#f7f8fb;border-radius:8px;padding:12px}}details{{margin:16px 0;background:#f8fafc;border-radius:8px;padding:10px 12px}}summary{{cursor:pointer;font-weight:700}}dl{{display:grid;grid-template-columns:minmax(130px,32%) 1fr;margin:0}}dt{{font-weight:700;text-transform:capitalize;color:#536078;padding:3px 8px 3px 0}}dd{{margin:0;padding:3px 0;overflow-wrap:anywhere}}ul{{padding-left:20px;margin:4px 0}}li{{margin:5px 0}}.decisions{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.target-decisions{{grid-template-columns:minmax(0,1fr)}}.decisions section{{border:1px solid #dce2ec;border-radius:8px;padding:10px}}.ground{{border-color:#8da0c2!important;background:#f4f7ff}}.miss{{border-color:#efaaa2!important;background:#fff7f5}}.match{{border-color:#91c8a0!important;background:#f5fcf6}}.review.good{{border-color:#91c8a0!important;background:#f5fcf6}}.review:not(.good){{border-color:#efc06c!important;background:#fffaf0}}.badge{{display:inline-block;padding:2px 7px;border-radius:99px;color:white;font-weight:750;font-size:12px}}.notify{{background:#c6332b}}.digest{{background:#9a6900}}.mute{{background:#586171}}.type,.confidence{{font-size:12px;font-weight:700;color:#44516a;margin-left:5px}}.correction{{display:flex;align-items:end;flex-wrap:wrap;gap:10px;margin-top:14px;padding-top:12px;border-top:1px solid #dce2ec}}.correction label{{display:grid;gap:3px;font-size:12px;font-weight:700;color:#44516a}}.correction select{{background:#fff;border:1px solid #aab5c9;border-radius:5px;padding:5px}}.correction small{{color:#58657a}}.view{{display:none}}.view.active{{display:block}}@media(max-width:900px){{.decisions{{grid-template-columns:1fr 1fr}}}}@media(max-width:600px){{.metrics,.decisions{{grid-template-columns:1fr}}main{{padding:16px 10px}}article{{padding:14px}}}}
      </style></head><body><main><h1>Router review</h1>{metrics}<div class="mode-control"><label>Sample + ground truth <input id="mode" type="range" min="0" max="1" step="1" value="0" aria-label="Review mode"> Test predictions</label></div><section id="sample" class="view active"><p class="subtitle">30 labeled samples: ground truth compared with Routers 0–3. A prediction matches only when both action and message type match.</p>{''.join(cards)}</section><section id="target" class="view"><p class="subtitle">Router 3 predictions from output.csv for the {len(target_cards)}-message test dataset. Choose a correction to save it automatically.</p>{''.join(target_cards)}</section></main><script>const mode=document.querySelector('#mode'),sample=document.querySelector('#sample'),target=document.querySelector('#target');mode.addEventListener('input',()=>{{const test=mode.value==='1';sample.classList.toggle('active',!test);target.classList.toggle('active',test)}});const fields=card=>Object.fromEntries([...card.querySelectorAll('select')].map(select=>[select.dataset.field,select.value]));const setStatus=(card,text)=>card.querySelector('[data-status]').textContent=text;async function save(card){{const values=fields(card),body={{message_id:card.dataset.messageId,action:values.action===card.dataset.originalAction?null:values.action,message_type:values.message_type===card.dataset.originalType?null:values.message_type}};setStatus(card,'Saving…');try{{const response=await fetch('/api/corrections',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}});if(!response.ok)throw new Error();setStatus(card,body.action||body.message_type?'Saved correction':'Original prediction')}}catch(error){{setStatus(card,'Start code/review_server.py to save corrections')}}}}document.querySelectorAll('.correction').forEach(card=>{{card.querySelectorAll('select').forEach(select=>select.addEventListener('change',()=>save(card)));}});fetch('/api/corrections').then(response=>response.ok?response.json():null).then(data=>{{for(const correction of data?.corrections||[]){{const card=document.querySelector(`.correction[data-message-id="${{correction.message_id}}"]`);if(!card)continue;for(const field of ['action','message_type'])if(correction[field])card.querySelector(`[data-field="${{field}}"]`).value=correction[field];setStatus(card,'Saved correction')}}}}).catch(()=>{{}});</script></body></html>''', encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
