import streamlit as st
import pandas as pd
import json
import time
import math
import os
import re
from typing import Literal
from pydantic import BaseModel, Field
import ollama

SUBMISSIONS_DIR = "submissions_1b"
CSV_FILENAME = "fop_grades_1b.csv"
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# =====================================================================
# Layer 2 Pydantic Checklist Models (BSCS-1B Rubric)
# =====================================================================

class Q1Checklist(BaseModel):
    input_and_loop: Literal["full", "partial", "none"] = Field(
        ..., description="reads positive integer and loops from 1 to N"
    )
    find_display_factors: Literal["full", "partial", "none"] = Field(
        ..., description="checks n % i == 0 and displays all factors on one line"
    )
    count_and_sum: Literal["full", "partial", "none"] = Field(
        ..., description="counts total factors, sums all factors, and displays both"
    )
    prime_composite: Literal["full", "partial", "none"] = Field(
        ..., description="checks if count == 2 to display Prime or Composite"
    )
    comments_output: Literal["full", "partial", "none"] = Field(
        ..., description="clear output present"
    )

class Q2Checklist(BaseModel):
    menu_input: Literal["full", "partial", "none"] = Field(
        ..., description="shows 4 room categories and reads category + number of nights"
    )
    rate_selection: Literal["full", "partial", "none"] = Field(
        ..., description="selects 4000/7500/12000/25000 via switch/if-else and handles invalid choice"
    )
    room_charges: Literal["full", "partial", "none"] = Field(
        ..., description="calculates room charges = nights * rate"
    )
    discount: Literal["full", "partial", "none"] = Field(
        ..., description="applies 15% discount if nights > 5 (partial for >= 5)"
    )
    tax_calc: Literal["full", "partial", "none"] = Field(
        ..., description="adds 8% tax on the discounted amount"
    )
    total_bill: Literal["full", "partial", "none"] = Field(
        ..., description="computes final total bill = discounted amount + tax"
    )
    output_comments: Literal["full", "partial", "none"] = Field(
        ..., description="displays all 6 bill details"
    )

class Q3Checklist(BaseModel):
    loop_and_pin: Literal["full", "partial", "none"] = Field(
        ..., description="fixed PIN 4321 and loop allowing up to 3 attempts"
    )
    access_granted: Literal["full", "partial", "none"] = Field(
        ..., description="checks PIN == 4321, displays Access Granted, and stops loop"
    )
    attempts_left: Literal["full", "partial", "none"] = Field(
        ..., description="on wrong PIN, displays remaining attempts"
    )
    card_blocked: Literal["full", "partial", "none"] = Field(
        ..., description="displays Card Blocked after 3 wrong attempts"
    )
    comments_clarity: Literal["full", "partial", "none"] = Field(
        ..., description="readable structure & clarity"
    )

class Q4Checklist(BaseModel):
    fix_for_semicolons: Literal["full", "partial", "none"] = Field(
        ..., description="changed commas to semicolons in for(i=1; i<=50; i++) (full if fixed + commented, partial if only fixed)"
    )
    fix_logical_and: Literal["full", "partial", "none"] = Field(
        ..., description="changed || to && in if(i%3==0 && i%5==0) (full if fixed + commented, partial if only fixed)"
    )
    fix_printf_comma: Literal["full", "partial", "none"] = Field(
        ..., description="added missing comma in printf(\"Count = %d\", count) (full if fixed + commented, partial if only fixed)"
    )

RUBRIC_CONFIG = {
    "Q1": {
        "model": Q1Checklist,
        "max": 6.0,
        "floor": 2.5,
        "weights": {
            "input_and_loop": 1.0,
            "find_display_factors": 1.5,
            "count_and_sum": 1.5,
            "prime_composite": 1.5,
            "comments_output": 0.5,
        },
        "labels": {
            "input_and_loop": "Input positive integer & for loop 1 to N (1.0)",
            "find_display_factors": "Find factors (n % i == 0) & display on one line (1.5)",
            "count_and_sum": "Count total factors & sum of all factors (1.5)",
            "prime_composite": "Display Prime (count == 2) or Composite (1.5)",
            "comments_output": "Clear output present (0.5)",
        }
    },
    "Q2": {
        "model": Q2Checklist,
        "max": 7.0,
        "floor": 2.5,
        "weights": {
            "menu_input": 1.0,
            "rate_selection": 1.5,
            "room_charges": 1.0,
            "discount": 1.0,
            "tax_calc": 1.0,
            "total_bill": 1.0,
            "output_comments": 0.5,
        },
        "labels": {
            "menu_input": "Show 4 room categories & read category + nights (1.0)",
            "rate_selection": "Select 4000/7500/12000/25000 & handle invalid choice (1.5)",
            "room_charges": "Room charges = nights * rate (1.0)",
            "discount": "15% discount if nights > 5 [partial if >= 5] (1.0)",
            "tax_calc": "8% tax on discounted amount (1.0)",
            "total_bill": "Total bill = discounted amount + tax (1.0)",
            "output_comments": "Display all 6 bill details (0.5)",
        }
    },
    "Q3": {
        "model": Q3Checklist,
        "max": 4.0,
        "floor": 1.5,
        "weights": {
            "loop_and_pin": 1.0,
            "access_granted": 1.0,
            "attempts_left": 1.0,
            "card_blocked": 0.5,
            "comments_clarity": 0.5,
        },
        "labels": {
            "loop_and_pin": "Fixed PIN 4321 & loop for max 3 attempts (1.0)",
            "access_granted": "Check PIN == 4321, print 'Access Granted' & stop (1.0)",
            "attempts_left": "Display attempts left on wrong PIN (1.0)",
            "card_blocked": "Display 'Card Blocked' after 3 wrong attempts (0.5)",
            "comments_clarity": "Readable structure & clarity (0.5)",
        }
    },
    "Q4": {
        "model": Q4Checklist,
        "max": 3.0,
        "floor": 1.5,
        "weights": {
            "fix_for_semicolons": 1.0,
            "fix_logical_and": 1.0,
            "fix_printf_comma": 1.0,
        },
        "labels": {
            "fix_for_semicolons": "Fix commas to semicolons in for(i=1; i<=50; i++) + comment (1.0)",
            "fix_logical_and": "Change || to && for divisible by both 3 and 5 + comment (1.0)",
            "fix_printf_comma": "Add missing comma in printf(\"Count = %d\", count) + comment (1.0)",
        }
    }
}

# =====================================================================
# Layer 1: Deterministic Python Splitter & Topic Matcher (BSCS-1B)
# =====================================================================

def split_and_classify(raw_text: str) -> dict[str, str]:
    """Split student submission into separate C blocks and match by BSCS-1B topic keywords."""
    if not raw_text or not raw_text.strip():
        return {"Q1": "", "Q2": "", "Q3": "", "Q4": ""}

    # 1. Identify split boundaries: explicit headers or main boundaries
    header_pattern = re.compile(
        r'(?im)^[ \t]*(?://|/\*|[#*])?[ \t]*(?:question|task|problem|code|q)[ \t]*[-_:#.]*[ \t]*([1-4])\b'
    )
    header_matches = list(header_pattern.finditer(raw_text))

    blocks: list[str] = []
    if len(header_matches) >= 2:
        for i, match in enumerate(header_matches):
            start = match.start()
            end = header_matches[i + 1].start() if i + 1 < len(header_matches) else len(raw_text)
            block_code = raw_text[start:end].strip()
            if block_code:
                blocks.append(block_code)
    else:
        main_pattern = re.compile(r'(?im)^[ \t]*(?:int|void)\s+main\s*\(', re.MULTILINE)
        main_matches = list(main_pattern.finditer(raw_text))
        if len(main_matches) >= 2:
            split_indices = [0]
            for i in range(1, len(main_matches)):
                pre_text = raw_text[main_matches[i - 1].start():main_matches[i].start()]
                last_brace = pre_text.rfind('}')
                if last_brace != -1:
                    split_idx = main_matches[i - 1].start() + last_brace + 1
                else:
                    inc = raw_text.rfind('#include', main_matches[i - 1].start(), main_matches[i].start())
                    split_idx = inc if inc != -1 else main_matches[i].start()
                split_indices.append(split_idx)
            split_indices.append(len(raw_text))

            for i in range(len(split_indices) - 1):
                block = raw_text[split_indices[i]:split_indices[i + 1]].strip()
                if block:
                    blocks.append(block)
        else:
            cleaned = raw_text.strip()
            if cleaned:
                blocks.append(cleaned)

    # 2. Score each block against BSCS-1B topic keywords
    topic_keywords = {
        "Q1": ["factor", "factors", "prime", "composite", "sum", "% i", "%i"],
        "Q2": ["hotel", "standard", "deluxe", "executive", "presidential", "4000", "7500", "12000", "25000", "nights", "tax", "room"],
        "Q3": ["4321", "pin", "atm", "access granted", "card blocked", "attempt", "attempts", "blocked"],
        "Q4": ["50", "i <= 50", "i<=50", "count = %d", "count=%d", "% 3", "%3", "% 5", "%5", "semicolon", "comma"],
    }

    assigned = {"Q1": "", "Q2": "", "Q3": "", "Q4": ""}
    candidate_matches = []

    for b_idx, block in enumerate(blocks):
        b_lower = block.lower()
        for q_key, kws in topic_keywords.items():
            kw_score = 0.0
            for kw in kws:
                if kw in b_lower:
                    kw_score += 1.5 if len(kw) > 4 else 1.0

            q_num = q_key[-1]
            if re.search(rf'(?im)\b(?:q|question|code)\s*[-_:#.]*\s*{q_num}\b', block[:250]):
                kw_score += 5.0

            if kw_score > 0:
                candidate_matches.append((kw_score, b_idx, q_key))

    candidate_matches.sort(key=lambda x: x[0], reverse=True)
    assigned_blocks = set()
    assigned_questions = set()

    for score, b_idx, q_key in candidate_matches:
        if b_idx not in assigned_blocks and q_key not in assigned_questions:
            assigned[q_key] = blocks[b_idx]
            assigned_blocks.add(b_idx)
            assigned_questions.add(q_key)

    return assigned

# =====================================================================
# Layer 3: Deterministic Score Calculation & Guardrails
# =====================================================================

def has_real_comments(code: str) -> bool:
    """Check if code contains user comments after stripping default Dev-C++ boilerplate."""
    if not code:
        return False
    # Strip Dev-C++ default console pauser comment block
    cleaned = re.sub(r'(?is)/\*.*?(?:console pauser|run this program).*?\*/', '', code)
    # Check if // or /* still exists in the code
    return bool(re.search(r'//|/\*', cleaned))

def apply_guardrails(q_key: str, checklist_data: dict, q_code: str) -> dict:
    """Apply deterministic Python guardrails to prevent LLM generosity bias."""
    if not checklist_data or not q_code:
        return checklist_data

    # Guardrails apply ONLY to Q4 (comments are ignored in Q1, Q2, Q3)
    if q_key == "Q4":
        # Guardrail 1: Ignore Dev-C++ boilerplate comments, require real comments for Q4 full marks
        if not has_real_comments(q_code):
            for field in ["fix_for_semicolons", "fix_logical_and", "fix_printf_comma"]:
                if field in checklist_data and str(checklist_data[field]).strip().lower() == "full":
                    checklist_data[field] = "partial"

        # Guardrail 2: Q4 '&&' hard check: if "&&" not present in q_code, force fix_logical_and to "none"
        if "&&" not in q_code:
            checklist_data["fix_logical_and"] = "none"

    return checklist_data

def calculate_question_score(q_key: str, checklist_data: dict | None, detected: bool, q_code: str = "") -> tuple[float, list[dict]]:
    """Convert full/partial/none checklist into numerical mark, applying rounding and floor."""
    if not detected or not checklist_data:
        return 0.0, []

    if q_code:
        apply_guardrails(q_key, checklist_data, q_code)

    cfg = RUBRIC_CONFIG[q_key]
    weights = cfg["weights"]
    labels = cfg["labels"]

    raw_sum = 0.0
    breakdown = []

    for item_key, weight in weights.items():
        raw_val = checklist_data.get(item_key, "none")
        status = str(raw_val).strip().lower() if raw_val is not None else "none"
        if status == "full":
            awarded = 1.0 * weight
        elif status == "partial":
            awarded = 0.5 * weight
        else:
            awarded = 0.0
        raw_sum += awarded
        breakdown.append({
            "Criterion": labels.get(item_key, item_key),
            "Status": status.upper(),
            "Weight": weight,
            "Awarded": awarded,
        })

    rounded = round(raw_sum * 4) / 4.0
    floored = max(cfg["floor"], rounded)
    final_score = min(cfg["max"], floored)

    return final_score, breakdown

# =====================================================================
# Grading Pipeline Runner
# =====================================================================

def run_grading_pipeline(student_id: str, student_name: str, code_text: str, selected_model: str) -> dict:
    """Execute Layer 1 classification and Layer 2 Ollama checklist evaluation."""
    classified = split_and_classify(code_text)

    system_prompt = (
        "You are an expert C code exam grader for BSCS-1B. "
        "Evaluate ONLY the provided C code against the exact rubric criteria checklist.\n"
        "Ignore minor C syntax typos (missing semicolons, missing `&` in scanf, `Case1:`, wrong format specifiers, undeclared variables).\n"
        "For Q1, Q2, and Q3, completely IGNORE missing comments—do NOT deduct marks if the student wrote no comments. "
        "ONLY require comments in Q4 (in Q4, award 'full' if a bug is fixed AND commented, or 'partial' if fixed without a comment).\n"
        "Return lowercase values only: \"full\", \"partial\", or \"none\".\n"
        "Reserve `full` ONLY when the logic for that criterion is genuinely correct. "
        "Use `partial` when the student attempted the criterion but has a clear logic bug "
        "(e.g., using `>= 5` instead of `> 5`, applying tax before discount, missing `break` after Access Granted, or fixing a Q4 bug without commenting it).\n"
        "If the student's code clearly implements the requirement correctly (even with different variable names), mark it \"full\".\n"
        "Reward clear logical intent. Return ONLY JSON matching the requested schema."
    )

    results = {}
    start_time = time.time()

    for q_key in ["Q1", "Q2", "Q3", "Q4"]:
        q_code = classified.get(q_key, "")
        if not q_code.strip():
            results[q_key] = {
                "detected": False,
                "checklist": None,
                "score": 0.0,
                "breakdown": [],
                "code": "",
                "error": None,
            }
        else:
            with st.spinner(f"Evaluating {q_key} against BSCS-1B rubric with {selected_model}..."):
                cfg = RUBRIC_CONFIG[q_key]
                checklist_model = cfg["model"]

                rubric_guidelines = []
                for field_name, label in cfg["labels"].items():
                    weight = cfg["weights"][field_name]
                    rubric_guidelines.append(f"- Field `{field_name}` (max {weight} pts): {label}")
                guidelines_text = "\n".join(rubric_guidelines)

                user_prompt = (
                    f"Examine this student's C code submission for {q_key}.\n\n"
                    f"CRITERIA CHECKLIST (evaluate every single field below):\n{guidelines_text}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"- Return lowercase values only: \"full\", \"partial\", or \"none\".\n"
                    f"- For Q1, Q2, and Q3, completely IGNORE missing comments—do NOT deduct marks if the student wrote no comments.\n"
                    f"- ONLY require comments in Q4 (in Q4, award 'full' if a bug is fixed AND commented, or 'partial' if fixed without a comment).\n"
                    f"- Reserve `full` ONLY when the logic for that criterion is genuinely correct.\n"
                    f"- Use `partial` when the student attempted the criterion but has a clear logic bug.\n"
                    f"- If the student's code clearly implements the requirement correctly (even with different variable names), mark it \"full\".\n\n"
                    f"STUDENT C CODE:\n```c\n{q_code}\n```"
                )

                content = ""
                has_error = False
                error_msg = ""
                try:
                    response = ollama.chat(
                        model=selected_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        keep_alive="60m",
                        options={"temperature": 0, "num_predict": 512, "num_ctx": 4096},
                        format=checklist_model.model_json_schema(),
                    )
                    content = response.get("message", {}).get("content", "")
                    parsed = json.loads(content)
                    validated = checklist_model(**parsed).model_dump()
                except Exception as e:
                    has_error = True
                    error_msg = str(e)
                    st.error(
                        f"❌ Error evaluating {q_key} via Ollama:\n\n"
                        f"**Exception:** `{error_msg}`\n\n"
                        f"**Raw Model Response:**\n```\n{content}\n```"
                    )
                    validated = {k: "none" for k in cfg["weights"].keys()}

                apply_guardrails(q_key, validated, q_code)
                score, breakdown = calculate_question_score(q_key, validated, detected=True, q_code=q_code)
                results[q_key] = {
                    "detected": True,
                    "checklist": validated,
                    "score": score,
                    "breakdown": breakdown,
                    "code": q_code,
                    "error": error_msg if has_error else None,
                    "raw_response": content if has_error else None,
                }

    total_time = time.time() - start_time
    return {
        "student_id": student_id,
        "student_name": student_name,
        "raw_code": code_text,
        "results": results,
        "time": total_time,
    }

# =====================================================================
# CSV & File Helpers
# =====================================================================

def sanitize_id(student_id: str) -> str:
    clean = re.sub(r'[^a-zA-Z0-9_\-]', '_', student_id.strip())
    return clean or "unknown"

def load_or_create_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["Student_ID", "Student_Name", "Q1", "Q2", "Q3", "Q4", "Total"])
        df.to_csv(path, index=False)
    return df

def update_csv(df: pd.DataFrame, student_id: str, student_name: str, scores: dict, total: int, path: str):
    if student_id in df["Student_ID"].astype(str).values:
        df.loc[df["Student_ID"].astype(str) == student_id, ["Student_Name", "Q1", "Q2", "Q3", "Q4", "Total"]] = [
            student_name, scores["Q1"], scores["Q2"], scores["Q3"], scores["Q4"], total
        ]
        updated = True
    else:
        new_row = {
            "Student_ID": student_id,
            "Student_Name": student_name,
            "Q1": scores["Q1"],
            "Q2": scores["Q2"],
            "Q3": scores["Q3"],
            "Q4": scores["Q4"],
            "Total": total,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        updated = False
    df.to_csv(path, index=False)
    return df, updated

def clear_inputs():
    st.session_state["input_student_id"] = ""
    st.session_state["input_student_name"] = ""
    st.session_state["input_code"] = ""
    if "graded_data" in st.session_state:
        del st.session_state["graded_data"]

def get_saved_student_ids() -> list[str]:
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    saved_files = {f[:-4] for f in os.listdir(SUBMISSIONS_DIR) if f.endswith(".txt")}

    csv_path = os.path.join(os.getcwd(), CSV_FILENAME)
    ids: list[str] = []
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for sid in df["Student_ID"].astype(str):
                safe = sanitize_id(sid)
                if safe in saved_files and sid not in ids:
                    ids.append(sid)
        except Exception:
            pass

    for f in sorted(saved_files):
        if f not in [sanitize_id(i) for i in ids]:
            ids.append(f)

    return ids

# =====================================================================
# Streamlit UI
# =====================================================================

st.set_page_config(page_title="C Exam Grader — BSCS-1B", layout="wide")
st.title("Fast Local C‑Language Exam Grader (BSCS‑1B)")
st.caption("3-Layer Hybrid Architecture: Deterministic Splitter → Per-Question Rubric Checklist → Strict Math Enforcement")

# ---------------------------------------------------------------------
# Safe Session State Handlers (MUST run before any widget instantiation)
# ---------------------------------------------------------------------

if st.session_state.get("needs_clear"):
    clear_inputs()
    st.session_state["needs_clear"] = False

if st.session_state.get("load_student_id"):
    target_id = st.session_state["load_student_id"]
    safe_id = sanitize_id(target_id)
    file_path = os.path.join(SUBMISSIONS_DIR, f"{safe_id}.txt")

    loaded_code = ""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            loaded_code = f.read()

    loaded_name = ""
    csv_path = os.path.join(os.getcwd(), CSV_FILENAME)
    if os.path.exists(csv_path):
        try:
            df_lookup = pd.read_csv(csv_path)
            matched = df_lookup[df_lookup["Student_ID"].astype(str) == str(target_id)]
            if not matched.empty and "Student_Name" in matched.columns and pd.notna(matched["Student_Name"].iloc[0]):
                loaded_name = str(matched["Student_Name"].iloc[0])
        except Exception:
            pass

    st.session_state["input_student_id"] = target_id
    st.session_state["input_student_name"] = loaded_name
    st.session_state["input_code"] = loaded_code
    st.session_state["load_student_id"] = None

# Sidebar model selector
model_options = {
    "qwen2.5-coder:7b (default)": "qwen2.5-coder:7b",
    "qwen2.5-coder:3b": "qwen2.5-coder:3b",
}
selected_label = st.sidebar.selectbox("Select Model", list(model_options.keys()))
selected_model = model_options[selected_label]

if "last_saved" in st.session_state:
    saved = st.session_state["last_saved"]
    action = "Updated" if saved["updated"] else "Saved"
    st.success(
        f"✅ Successfully {action} grades for Student ID: **{saved['student_id']}** "
        f"(Total: **{saved['total']}/20**) into `{CSV_FILENAME}` and saved submission code to `{SUBMISSIONS_DIR}/`!"
    )
    del st.session_state["last_saved"]

if "input_student_id" not in st.session_state:
    st.session_state["input_student_id"] = ""
if "input_student_name" not in st.session_state:
    st.session_state["input_student_name"] = ""
if "input_code" not in st.session_state:
    st.session_state["input_code"] = ""

student_id = st.text_input("Student ID", key="input_student_id", help="Required")
student_name = st.text_input("Student Name (optional)", key="input_student_name")
code_text = st.text_area("Complete C Code (Q1–Q4)", height=260, key="input_code")

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    grade_btn = st.button("Grade Submission", type="primary")
with col_btn2:
    clear_btn = st.button("Clear Form", on_click=clear_inputs)

# ---------------------------------------------------------------------
# Execution: Layer 1 + Layer 2 Grading
# ---------------------------------------------------------------------
should_grade = grade_btn or st.session_state.get("auto_grade", False)

if should_grade:
    if st.session_state.get("auto_grade"):
        st.session_state["auto_grade"] = False

    if not student_id.strip():
        st.error("Student ID is required.")
    elif not code_text.strip():
        st.error("Please enter C code to grade.")
    else:
        st.session_state["graded_data"] = run_grading_pipeline(
            student_id=student_id,
            student_name=student_name,
            code_text=code_text,
            selected_model=selected_model,
        )
        st.rerun()

# ---------------------------------------------------------------------
# Review-Before-Save UI (Layer 3)
# ---------------------------------------------------------------------
if "graded_data" in st.session_state:
    graded = st.session_state["graded_data"]
    results = graded["results"]

    st.markdown("---")
    st.subheader(f"Grading Review for Student: {graded['student_id']}")
    st.caption(f"Inference Time: {graded['time']:.2f} seconds")

    missing_qs = [q for q in ["Q1", "Q2", "Q3", "Q4"] if not results[q]["detected"]]
    if missing_qs:
        st.warning(
            f"⚠️ **Warning**: {', '.join(missing_qs)} was not detected in the submission "
            f"(marked as Missing / Not Attempted, scored 0.0). Please double-check if the student submitted it."
        )

    for q_key in ["Q1", "Q2", "Q3", "Q4"]:
        q_res = results[q_key]
        status_str = f"✅ Detected ({q_res['score']:.2f}/{RUBRIC_CONFIG[q_key]['max']})" if q_res["detected"] else "❌ Missing (0.0)"
        with st.expander(f"{q_key} Rubric Breakdown — {status_str}", expanded=False):
            if q_res.get("error"):
                st.error(f"⚠️ Parsing/Ollama Error for {q_key}: {q_res['error']}\n\n**Raw Response:**\n```\n{q_res.get('raw_response', '')}\n```")
            if q_res["detected"] and q_res["breakdown"]:
                df_bk = pd.DataFrame(q_res["breakdown"])
                st.dataframe(df_bk, use_container_width=True)
            else:
                st.info(f"{q_key} was not detected by Layer 1 keywords. Assigned 0.0.")
            if q_res["code"]:
                st.markdown("**Matched C Code:**")
                st.code(q_res["code"], language="c")

    st.markdown("### Review & Override Marks")
    st.caption("Marks are pre-filled based on the rubric checklist. You can edit any question mark before saving.")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        val_q1 = st.number_input(
            "Q1 (Max 6.0)", min_value=0.0, max_value=6.0, step=0.25,
            value=float(results["Q1"]["score"]), key="score_q1"
        )
    with col2:
        val_q2 = st.number_input(
            "Q2 (Max 7.0)", min_value=0.0, max_value=7.0, step=0.25,
            value=float(results["Q2"]["score"]), key="score_q2"
        )
    with col3:
        val_q3 = st.number_input(
            "Q3 (Max 4.0)", min_value=0.0, max_value=4.0, step=0.25,
            value=float(results["Q3"]["score"]), key="score_q3"
        )
    with col4:
        val_q4 = st.number_input(
            "Q4 (Max 3.0)", min_value=0.0, max_value=3.0, step=0.25,
            value=float(results["Q4"]["score"]), key="score_q4"
        )
    with col5:
        computed_total = min(20, math.ceil(val_q1 + val_q2 + val_q3 + val_q4))
        st.metric("Total Score", f"{computed_total}/20")

    if st.button("Save to CSV", type="primary"):
        csv_path = os.path.join(os.getcwd(), CSV_FILENAME)
        df_csv = load_or_create_csv(csv_path)
        scores_to_save = {"Q1": val_q1, "Q2": val_q2, "Q3": val_q3, "Q4": val_q4}
        df_csv, updated = update_csv(
            df_csv, graded["student_id"], graded["student_name"], scores_to_save, computed_total, csv_path
        )

        os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
        safe_id = sanitize_id(graded["student_id"])
        file_path = os.path.join(SUBMISSIONS_DIR, f"{safe_id}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(graded.get("raw_code", ""))

        st.session_state["last_saved_id"] = graded["student_id"]
        st.session_state["last_saved"] = {
            "student_id": graded["student_id"],
            "total": computed_total,
            "updated": updated,
        }
        del st.session_state["graded_data"]
        st.session_state["needs_clear"] = True
        st.rerun()

# ---------------------------------------------------------------------
# Re-evaluate Previous Saved Submissions Controls
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📂 Re-evaluate Saved Submission (BSCS-1B)")
saved_ids = get_saved_student_ids()

if saved_ids:
    default_idx = 0
    last_id = st.session_state.get("last_saved_id")
    if last_id and last_id in saved_ids:
        default_idx = saved_ids.index(last_id)

    col_re1, col_re2 = st.columns([3, 1])
    with col_re1:
        selected_re_id = st.selectbox(
            "Select a saved student submission to re-evaluate",
            options=saved_ids,
            index=default_idx,
            key="re_eval_select",
        )
    with col_re2:
        st.write("")
        if st.button("Load & Re-evaluate", type="secondary"):
            st.session_state["load_student_id"] = selected_re_id
            st.session_state["auto_grade"] = True
            st.rerun()
else:
    st.info(f"No saved submissions found in `{SUBMISSIONS_DIR}/` yet. Submissions are saved automatically when clicking 'Save to CSV'.")

# ---------------------------------------------------------------------
# Live CSV Table & Export
# ---------------------------------------------------------------------
st.markdown("---")
csv_path = os.path.join(os.getcwd(), CSV_FILENAME)
df_display = load_or_create_csv(csv_path)
st.subheader(f"Live Grades Table (`{CSV_FILENAME}`)")
st.dataframe(df_display, use_container_width=True)

csv_bytes = df_display.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv_bytes,
    file_name=CSV_FILENAME,
    mime="text/csv",
)