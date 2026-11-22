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

SUBMISSIONS_DIR = "submissions"
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# =====================================================================
# Layer 2 Pydantic Checklist Models (Rubric-Driven)
# =====================================================================

class Q1Checklist(BaseModel):
    input_and_loop: Literal["full", "partial", "none"] = Field(
        ..., description="reads N and loops over entries"
    )
    validation: Literal["full", "partial", "none"] = Field(
        ..., description="checks <0 or >100, shows error, avoids counting invalid entry"
    )
    high_low: Literal["full", "partial", "none"] = Field(
        ..., description="tracks highest and lowest marks"
    )
    average: Literal["full", "partial", "none"] = Field(
        ..., description="computes sum / count (partial for int division or dividing by N)"
    )
    pass_stats: Literal["full", "partial", "none"] = Field(
        ..., description="counts marks >= 50 and computes pass percentage"
    )
    comments_output: Literal["full", "partial", "none"] = Field(
        ..., description="comments or clear output present"
    )

class Q2Checklist(BaseModel):
    menu_input: Literal["full", "partial", "none"] = Field(
        ..., description="shows 4 packages and reads type + GB"
    )
    rate_selection: Literal["full", "partial", "none"] = Field(
        ..., description="selects 30/25/20/15 via switch/if-else"
    )
    price_calc: Literal["full", "partial", "none"] = Field(
        ..., description="multiplies GB * rate"
    )
    discount: Literal["full", "partial", "none"] = Field(
        ..., description="applies 100 discount for >800 (partial for >=800)"
    )
    service_charge: Literal["full", "partial", "none"] = Field(
        ..., description="adds 50 for Business Pack only"
    )
    final_payable: Literal["full", "partial", "none"] = Field(
        ..., description="computes price - discount + service charge"
    )
    output_comments: Literal["full", "partial", "none"] = Field(
        ..., description="displays outputs/comments"
    )

class Q3Checklist(BaseModel):
    loop_1_to_30: Literal["full", "partial", "none"] = Field(
        ..., description="loops from 1 to 30"
    )
    divisibility: Literal["full", "partial", "none"] = Field(
        ..., description="checks 3&5 (ThreeFive), 3 (Three), 5 (Five), else number"
    )
    output_format: Literal["full", "partial", "none"] = Field(
        ..., description="prints outputs clearly"
    )
    sum_multiples: Literal["full", "partial", "none"] = Field(
        ..., description="accumulates and prints sum of numbers divisible by 3 or 5"
    )
    clarity: Literal["full", "partial", "none"] = Field(
        ..., description="readable structure"
    )

class Q4Checklist(BaseModel):
    fix_semicolon: Literal["full", "partial", "none"] = Field(
        ..., description="added ; after sum=0 (full if fixed + commented, partial if only fixed)"
    )
    fix_loop_bound: Literal["full", "partial", "none"] = Field(
        ..., description="changed i < n to i <= n (full if fixed + commented, partial if only fixed)"
    )
    fix_equality: Literal["full", "partial", "none"] = Field(
        ..., description="changed i % 2 = 0 to i % 2 == 0 (full if fixed + commented, partial if only fixed)"
    )

RUBRIC_CONFIG = {
    "Q1": {
        "model": Q1Checklist,
        "max": 6.0,
        "floor": 2.5,
        "weights": {
            "input_and_loop": 1.0,
            "validation": 1.5,
            "high_low": 1.0,
            "average": 1.0,
            "pass_stats": 1.0,
            "comments_output": 0.5,
        },
        "labels": {
            "input_and_loop": "Input N & loop over entries (1.0)",
            "validation": "Validate marks <0 or >100, error message, don't count invalid (1.5)",
            "high_low": "Track highest & lowest marks (1.0)",
            "average": "Compute average = sum / valid_count (1.0)",
            "pass_stats": "Pass count (>=50) & pass percentage (1.0)",
            "comments_output": "Comments & clear output (0.5)",
        }
    },
    "Q2": {
        "model": Q2Checklist,
        "max": 7.0,
        "floor": 2.5,
        "weights": {
            "menu_input": 1.0,
            "rate_selection": 1.5,
            "price_calc": 1.0,
            "discount": 1.0,
            "service_charge": 1.0,
            "final_payable": 1.0,
            "output_comments": 0.5,
        },
        "labels": {
            "menu_input": "Show 4 packages menu & read type + GB (1.0)",
            "rate_selection": "Select rate 30/25/20/15 & handle invalid choice (1.5)",
            "price_calc": "Price = GB * rate (1.0)",
            "discount": "Rs. 100 discount if price > 800 (1.0)",
            "service_charge": "Rs. 50 service charge for Business Pack only (1.0)",
            "final_payable": "Final payable = price - discount + service charge (1.0)",
            "output_comments": "Display outputs & comments (0.5)",
        }
    },
    "Q3": {
        "model": Q3Checklist,
        "max": 4.0,
        "floor": 1.5,
        "weights": {
            "loop_1_to_30": 0.5,
            "divisibility": 1.5,
            "output_format": 0.5,
            "sum_multiples": 1.0,
            "clarity": 0.5,
        },
        "labels": {
            "loop_1_to_30": "For loop from 1 to 30 (0.5)",
            "divisibility": "Divisibility: 3&5 'ThreeFive', 3 'Three', 5 'Five', else number (1.5)",
            "output_format": "One line space-separated output format (0.5)",
            "sum_multiples": "Sum of numbers divisible by 3 or 5 [expected 225] (1.0)",
            "clarity": "Compilation clarity & readable structure (0.5)",
        }
    },
    "Q4": {
        "model": Q4Checklist,
        "max": 3.0,
        "floor": 1.5,
        "weights": {
            "fix_semicolon": 1.0,
            "fix_loop_bound": 1.0,
            "fix_equality": 1.0,
        },
        "labels": {
            "fix_semicolon": "Fix missing semicolon after sum = 0 (1.0)",
            "fix_loop_bound": "Change i < n to i <= n (1.0)",
            "fix_equality": "Change i % 2 = 0 to i % 2 == 0 (1.0)",
        }
    }
}

# =====================================================================
# Layer 1: Deterministic Python Splitter & Topic Matcher
# =====================================================================

def split_and_classify(raw_text: str) -> dict[str, str]:
    """Split student submission into separate C blocks and match by topic keywords."""
    if not raw_text or not raw_text.strip():
        return {"Q1": "", "Q2": "", "Q3": "", "Q4": ""}

    # 1. Identify split boundaries: explicit headers or main boundaries
    header_pattern = re.compile(
        r'(?im)^[ \t]*(?://|/\*|[#*])?[ \t]*(?:question|task|problem|q)[ \t]*[-_:#.]*[ \t]*([1-4])\b'
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
        # Check for multiple 'main' declarations
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

    # 2. Score each block against topic keywords
    topic_keywords = {
        "Q1": ["highest", "lowest", "average", "pass", "invalid marks", "students", "marks"],
        "Q2": ["daily", "weekly", "monthly", "business", "package", "gb", "800", "discount", "service"],
        "Q3": ["threefive", "three", "five", "30", "%3", "% 3", "%5", "% 5"],
        "Q4": ["even", "sum of even", "i%2", "i % 2", "enter n", "missing", "semicolon"],
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

            # Boost if explicit header is near the top of block
            q_num = q_key[-1]
            if re.search(rf'(?im)\b(?:q|question)\s*[-_:#.]*\s*{q_num}\b', block[:250]):
                kw_score += 5.0

            if kw_score > 0:
                candidate_matches.append((kw_score, b_idx, q_key))

    # Greedily assign best non-overlapping 1-to-1 matches
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
# Layer 3: Deterministic Score Calculation
# =====================================================================

def calculate_question_score(q_key: str, checklist_data: dict | None, detected: bool) -> tuple[float, list[dict]]:
    """Convert full/partial/none checklist into numerical mark, applying rounding and floor."""
    if not detected or not checklist_data:
        return 0.0, []

    cfg = RUBRIC_CONFIG[q_key]
    weights = cfg["weights"]
    labels = cfg["labels"]

    raw_sum = 0.0
    breakdown = []

    for item_key, weight in weights.items():
        raw_val = checklist_data.get(item_key, "none")
        # Normalize case: trim and lowercase so FULL or Full correctly maps
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

    # Round to nearest 0.25
    rounded = round(raw_sum * 4) / 4.0
    # Enforce minimum floor only if detected
    floored = max(cfg["floor"], rounded)
    # Clamp to question max
    final_score = min(cfg["max"], floored)

    return final_score, breakdown

# =====================================================================
# Grading Pipeline Runner
# =====================================================================

def run_grading_pipeline(student_id: str, student_name: str, code_text: str, selected_model: str) -> dict:
    """Execute Layer 1 classification and Layer 2 Ollama checklist evaluation."""
    classified = split_and_classify(code_text)

    system_prompt = (
        "You are an expert C code exam grader. "
        "Evaluate ONLY the provided C code against the exact rubric criteria checklist.\n"
        "Ignore minor C syntax typos (missing semicolons, missing `&` in scanf, `Case1:`, wrong format specifiers, undeclared variables).\n"
        "Return lowercase values only: \"full\", \"partial\", or \"none\". "
        "If the student's code clearly implements the requirement (even with different variable names like `bill` instead of `price`), mark it \"full\".\n"
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
            with st.spinner(f"Evaluating {q_key} against rubric with {selected_model}..."):
                cfg = RUBRIC_CONFIG[q_key]
                checklist_model = cfg["model"]

                # Explicitly list each JSON field name alongside its plain-English rubric rule
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
                    f"- If the student's code clearly implements the requirement (even with different variable names like `bill` instead of `price`), mark it \"full\".\n\n"
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

                score, breakdown = calculate_question_score(q_key, validated, detected=True)
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
    """Sanitize student ID for use in filesystem paths."""
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
    """Safely reset input widget states."""
    st.session_state["input_student_id"] = ""
    st.session_state["input_student_name"] = ""
    st.session_state["input_code"] = ""
    if "graded_data" in st.session_state:
        del st.session_state["graded_data"]

def get_saved_student_ids() -> list[str]:
    """Return all Student_IDs that have a saved submission file in submissions/."""
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    saved_files = {f[:-4] for f in os.listdir(SUBMISSIONS_DIR) if f.endswith(".txt")}

    csv_path = os.path.join(os.getcwd(), "fop_grades.csv")
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

    # Also include any files not in CSV
    for f in sorted(saved_files):
        if f not in [sanitize_id(i) for i in ids]:
            ids.append(f)

    return ids

# =====================================================================
# Streamlit UI
# =====================================================================

st.set_page_config(page_title="C Exam Grader (3-Layer Hybrid)", layout="wide")
st.title("Fast Local C‑Language Exam Grader")
st.caption("3-Layer Hybrid Architecture: Deterministic Splitter → Per-Question Rubric Checklist → Strict Math Enforcement")

# ---------------------------------------------------------------------
# Safe Session State Handlers (MUST run before any widget instantiation)
# ---------------------------------------------------------------------

# 1. Reset inputs if requested after saving
if st.session_state.get("needs_clear"):
    clear_inputs()
    st.session_state["needs_clear"] = False

# 2. Load submission for re-evaluation if requested
if st.session_state.get("load_student_id"):
    target_id = st.session_state["load_student_id"]
    safe_id = sanitize_id(target_id)
    file_path = os.path.join(SUBMISSIONS_DIR, f"{safe_id}.txt")

    loaded_code = ""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            loaded_code = f.read()

    # Look up student name from CSV
    loaded_name = ""
    csv_path = os.path.join(os.getcwd(), "fop_grades.csv")
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

# Persistent save notice
if "last_saved" in st.session_state:
    saved = st.session_state["last_saved"]
    action = "Updated" if saved["updated"] else "Saved"
    st.success(
        f"✅ Successfully {action} grades for Student ID: **{saved['student_id']}** "
        f"(Total: **{saved['total']}/20**) into `fop_grades.csv` and saved submission code to `submissions/`!"
    )
    del st.session_state["last_saved"]

# Input fields
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
# Execution: Layer 1 + Layer 2 Grading (Manual or Auto-Re-evaluate)
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

    # Missing question warning banner
    missing_qs = [q for q in ["Q1", "Q2", "Q3", "Q4"] if not results[q]["detected"]]
    if missing_qs:
        st.warning(
            f"⚠️ **Warning**: {', '.join(missing_qs)} was not detected in the submission "
            f"(marked as Missing / Not Attempted, scored 0.0). Please double-check if the student submitted it."
        )

    # Expandable checklist breakdown for each question
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

    # Editable override inputs
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

    # Save to CSV and save submission file
    if st.button("Save to CSV", type="primary"):
        csv_path = os.path.join(os.getcwd(), "fop_grades.csv")
        df_csv = load_or_create_csv(csv_path)
        scores_to_save = {"Q1": val_q1, "Q2": val_q2, "Q3": val_q3, "Q4": val_q4}
        df_csv, updated = update_csv(
            df_csv, graded["student_id"], graded["student_name"], scores_to_save, computed_total, csv_path
        )

        # Save student raw code to submissions/{safe_id}.txt
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
st.subheader("📂 Re-evaluate Saved Submission")
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
        st.write("")  # align vertically with selectbox
        if st.button("Load & Re-evaluate", type="secondary"):
            st.session_state["load_student_id"] = selected_re_id
            st.session_state["auto_grade"] = True
            st.rerun()
else:
    st.info("No saved submissions found in `submissions/` yet. Submissions are saved automatically when clicking 'Save to CSV'.")

# ---------------------------------------------------------------------
# Live CSV Table & Export
# ---------------------------------------------------------------------
st.markdown("---")
csv_path = os.path.join(os.getcwd(), "fop_grades.csv")
df_display = load_or_create_csv(csv_path)
st.subheader("Live Grades Table (`fop_grades.csv`)")
st.dataframe(df_display, use_container_width=True)

csv_bytes = df_display.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv_bytes,
    file_name="fop_grades.csv",
    mime="text/csv",
)
