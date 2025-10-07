# app.py
from flask import Flask, render_template, request, jsonify, session, redirect
from dotenv import load_dotenv
import os
import requests
import logging
from functools import wraps
from groq import Groq
import re
import json
import time
from flask_session import Session

# Initialize environment and logging
load_dotenv()
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24).hex())
# Use filesystem-based sessions
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session_data'  # folder to store session files
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True  # signs the session cookie
Session(app)

# API Configuration
API_CONFIG = {
    "hr": {
        "key": os.getenv("GROQ_HR_API_KEY"),
        "model": "llama-3.1-8b-instant",
        "temperature": 0.7,
        "max_tokens": 500
    },
    "technical": {
        "key": os.getenv("GROQ_TECH_API_KEY"),
        "model": "llama-3.1-8b-instant",
        "temperature": 0.8,
        "max_tokens": 750
    },
    "cultural": {
        "key": os.getenv("GROQ_CULTURE_API_KEY"),
        "model": "llama-3.1-8b-instant",
        "temperature": 0.6,
        "max_tokens": 600
    },
    "report": {
        "key": os.getenv("GROQ_CULTURE_API_KEY"),
        "model": "llama-3.1-8b-instant",
        "temperature": 0.6,
        "max_tokens": 600
    }
}

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Enhanced interview prompts with scoring guidance
INTERVIEW_PROFILES = {
    "hr": {
        "instructions": """
        You are a professional HR interviewer. Conduct a realistic HR interview for the given job.
Ask only easy to medium-level behavioral and situational questions.
Do not ask more than 1 difficult question.
Ask one question at a time and wait for a response before proceeding.
Do not include outro messages and if the user wants to end the interview ask him to presss the end interview button.
Make sure you are on-point don't explain about the users status, go ahead as it is an original professional interview.
Do not mention user's performance in the interview and also don't give brief answer about the previous question.
Do not include bracketed content or hints (like [pause], [suggestion], etc.)
The assistant should behave like a serious professional interviewer.
If you feel that by question 8, 9, or earlier the candidate has already demonstrated enough in all the grading areas, you can gracefully conclude the interview early by thanking the candidate
Do not ever ask more than 15 questions. If a question is not answered well, you can probe once but don’t count that as a new question.
At the end of the interview (whether early or after 15 questions), clearly state the conclusion and ask the user to press the end interview button to generate report.

        Conduct a behavioral interview focusing on:
        - Teamwork (30% weight)
        - Problem-solving (30%)
        - Adaptability (20%)
        - Communication (20%)
        
        Evaluation Criteria:
        1. STAR format responses preferred
        -- Dont show percentages and score of the previous answer in the question and also if the user tells us that he/she dont know the answer for this just skip to the next question and dont generate too much of large question
        -- Never coding related questions as it is a theory based interview to test the in depth conceptual knowledge of the user, and never provide answer to the user just skip to the next question
        --- Interview ending should be provided in the seperate message.
        -- Don't Provide context to the user about the previous questions as it causes waste of time to the user.
        -- Don't repeat the questions
        """,
        "question_mix": {
            "teamwork": 2,
            "problem_solving": 1,
            "adaptability": 2,
            "communication": 2
        }

    },
    "technical": {
        "instructions": """
        Assess technical knowledge with:
        - Core concepts (40%)
        - Practical application (30%)
        - Problem analysis (20%)
        - Industry trends (10%)
        
        Evaluation Guidelines:
        1. Clarify ambiguous answers
        -- Dont show percentages and score of the previous answer in the question and also if the user tells us that he/she dont know the answer for this just skip to the next question and dont generate too much of large question
        -- Never coding related questions as it is a theory based interview to test the in depth conceptual knowledge of the user, and never provide answer to the user just skip to the next question
        --- Interview ending should be provided in the seperate message that is there should be a seperate outro message
        -- Don't Provide context to the user about the previous questions as it causes waste of time to the user.
        -- Don't repeat the questions
        -- Ask situation based, aptitute based questions by twisting concepts to test in depth knowledge of the user rather than asking normal questions like differences etc
        -- Do not include outro messages and if the user wants to end the interview ask him to presss the end interview button.
        -- Do not mention user's performance in the interview and also don't give brief answer about the previous question.
        -- Do not include bracketed content or hints (like [pause], [suggestion], etc.)
        -- The assistant should behave like a serious professional interviewer.
        -- If you feel that by question 8, 9, or earlier the candidate has already demonstrated enough in all the grading areas, you can gracefully conclude the interview early by thanking the candidate
        -- Do not ever ask more than 15 questions. If a question is not answered well, you can probe once but don’t count that as a new question.
        -- At the end of the interview (whether early or after 15 questions), clearly state the conclusion and ask the user to press the end interview button to generate report.

        """,
        "depth_levels": {
            "basic": 4,
            "intermediate": 4,
            "advanced": 2
        }
    },
    "cultural": {
        "instructions": """
        Evaluate cultural fit through:
        - Values alignment (40%)
        - Work style (30%)
        - Conflict resolution (20%)
        - Growth mindset (10%)
        
        Assessment Method:
        1. Look for specific examples
        -- Dont show percentages and score of the previous answer in the question and also if the user tells us that he/she dont know the answer for a particular question, just go for asking the next question and dont generate too much of large question
        -- Never ask coding related questions as it is a theory based interview to test the in depth conceptual knowledge of the user, and never provide answer to the user just skip to the next question
        --- Interview ending should be provided in the seperate message that is there should be a seperate outro message
        -- Don't Provide context to the user about the previous questions as it causes waste of time to the user.
        -- Don't repeat the questions
        -- Do not include outro messages and if the user wants to end the interview ask him to presss the end interview button.
        -- Do not mention user's performance in the interview and also don't give brief answer about the previous question.
        -- Do not include bracketed content or hints (like [pause], [suggestion], etc.)
        -- The assistant should behave like a serious professional interviewer.
        -- If you feel that by question 8, 9, or earlier the candidate has already demonstrated enough in all the grading areas, you can gracefully conclude the interview early by thanking the candidate
        -- Do not ever ask more than 15 questions. If a question is not answered well, you can probe once but don’t count that as a new question.
        -- At the end of the interview (whether early or after 15 questions), clearly state the conclusion and ask the user to press the end interview button to generate report.
        """
    },
    "report": {
    "instructions": """
You are an assistant generating a structured post-interview feedback report. 
Be concise, critical, and format the report **exactly** using the following sections and tags.

---
Strictly follow the below format

### SCORES ###
{
  "overall_score": 75,
  "accuracy": 82,
  "communication": 78,
  "confidence": "Medium",
  "recommendation": "Excellent candidate. Ready for the next round."
}
### END SCORES ###

---

### ANALYSIS ###
Provide a detailed paragraph summarizing the candidate’s performance, technical depth, communication, and team fit.

### END ANALYSIS ###

---

### STRENGTHS ###
Write 2–3 clear sentences (no bullet points, no symbols) summarizing the candidate’s key strengths.

### END STRENGTHS ###

---

### IMPROVEMENTS ###
Write 2–3 clear sentences (no bullet points, no symbols) highlighting areas the candidate should improve on.

### END IMPROVEMENTS ###

---

Make sure:
- Each section starts and ends with the EXACT tags.
- No extra content outside the required sections.
- Be brutally honest and critical with scores and feedback.
- Avoid generic phrases — tailor insights to the actual conversation.
- Ensure the scoring is consistent and realistic, reflecting how a human interviewer would rate candidates.
- Do not randomly give very high or very low scores.
- Only give high scores if the candidate demonstrates exceptional clarity, depth, and communication.
- Maintain a balanced, fair scoring pattern across all reports so candidates can be compared meaningfully.
- Do not inflate or deflate scores — judge based on the actual conversation content.

Now analyze this interview conversation and return the report in the above format.
"""
}
}

def validate_interview_type(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'job_data' not in session:
            app.logger.warning("No job data in session")
            return redirect('/')
        if session['job_data'].get('type') not in API_CONFIG:
            app.logger.error(f"Invalid interview type in session: {session['job_data'].get('type')}")
            session.clear()
            return redirect('/')
        return f(*args, **kwargs)
    return wrapper

@app.route('/')
def home():
    """Initialize a new interview session"""
    session.clear()
    return render_template('index.html')

@app.route('/save-job-data', methods=['POST'])
def save_job_data():
    """Store job information in session"""
    try:
        data = request.get_json()
        if not all(key in data for key in ['jobRole', 'jobDescription', 'interviewType']):
            raise ValueError("Missing required fields")
            
        session['job_data'] = {
            'role': data['jobRole'].strip(),
            'description': data['jobDescription'].strip(),
            'type': data['interviewType'].strip()
        }
        
        app.logger.info(f"New session started for {session['job_data']['role']} ({session['job_data']['type']})")
        return jsonify({"status": "success"})
        
    except Exception as e:
        app.logger.error(f"Data saving failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Invalid data provided",
            "error": str(e)
        }), 400

@app.route('/interview')
@validate_interview_type
def start_interview():
    """Initialize the interview conversation"""
    interview_type = session['job_data']['type']
    
    system_prompt = (
        f"{INTERVIEW_PROFILES[interview_type]['instructions']}\n"
        f"Position: {session['job_data']['role']}\n"
        f"Job Description:\n{session['job_data']['description']}"
    )
    
    session['conversation'] = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": f"Let's begin our {interview_type} discussion about {session['job_data']['role']}. Could you introduce yourself and your relevant experience?"}
    ]
    session['question_count'] = 0
    # Initialize the response_times list here
    session['response_times'] = []
    # Set the initial timestamp for the very first question
    session['last_ai_message_time'] = time.time()
    
    app.logger.debug(f"Interview initialized with system prompt: {system_prompt[:200]}...")
    return render_template('interview.html')

@app.route('/ask', methods=['POST'])
@validate_interview_type
def ask_question():
    """Process interview Q&A with AI interruption protection"""
    try:
        # Block if AI is still responding
        if session.get("is_ai_responding", False):
            return jsonify({
                "reply": "🚫 Please wait until the interviewer completes the question.",
                "status": "waiting"
            }), 429

        # Get user message
        user_message = request.form.get('message', '').strip()
        if not user_message:
            raise ValueError("Empty message received")

        # ** Calculate the user's response time **
        if 'last_ai_message_time' in session:
            user_response_time = time.time() - session['last_ai_message_time']
            # Initialize the list if it doesn't exist
            if 'response_times' not in session:
                session['response_times'] = []
            
            # Append the new response time
            session['response_times'].append(round(user_response_time))
            # Critical: Tell Flask the session has been modified
            session.modified = True

        # Mark AI as responding
        session["is_ai_responding"] = True
        
        # Update conversation
        interview_type = session['job_data']['type']
        session['conversation'].append({"role": "user", "content": user_message})
        session['question_count'] += 1
        session.modified = True

        # Prepare Groq API request
        config = API_CONFIG[interview_type]
        payload = {
            "model": config["model"],
            "messages": session['conversation'],
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"]
        }

        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {config['key']}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=20
        )
        response.raise_for_status()

        # Append AI reply
        ai_reply = response.json()['choices'][0]['message']['content']
        session['conversation'].append({"role": "assistant", "content": ai_reply})
        session["is_ai_responding"] = False
        session.modified = True
        
        # Store the current time for the next response calculation
        session['last_ai_message_time'] = time.time()

        return jsonify({
            "reply": ai_reply,
            "question_count": session['question_count'],
            "status": "success"
        })

    except requests.exceptions.RequestException as e:
        session["is_ai_responding"] = False
        app.logger.error(f"API request failed: {str(e)}")
        return jsonify({
            "reply": f"Our interview system is temporarily unavailable. Please try again shortly.{e}",
            "error": "api_error",
            "status": "error"
        }), 503

    except Exception as e:
        session["is_ai_responding"] = False
        app.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            "reply": "An unexpected error occurred. Please refresh the page and try again.",
            "error": "server_error",
            "status": "error"
        }), 500


@app.route("/generate-report", methods=["POST"])
def generate_report():
    try:
        full_conversation = session.get("conversation", [])
        if not full_conversation:
            return jsonify({"status": "error", "message": "No conversation found"}), 400

        # Filter only user and assistant messages
        filtered_messages = [msg for msg in full_conversation if msg['role'] in ['user', 'assistant']]

        # If conversation is too short, don't analyze
        user_responses = [msg for msg in filtered_messages if msg['role'] == 'user']
        if len(user_responses) < 2:  # adjust threshold as needed
            fallback_report = """\
### SCORES ###
{
  "overall_score": 0,
  "accuracy": 0,
  "communication": 0,
  "confidence": "Low",
  "recommendation": "Not enough responses to evaluate the candidate."
}
### END SCORES ###

---

### ANALYSIS ###
Not enough responses were provided by the candidate to generate a meaningful report.

### END ANALYSIS ###

---

### STRENGTHS ###
- Not Available

### END STRENGTHS ###

---

### IMPROVEMENTS ###
- Candidate should answer more questions to be properly assessed.

### END IMPROVEMENTS ###
"""
            session['latest_report'] = {
                "report_text": fallback_report,
                "report_score": {
                    "overall_score": 0,
                    "accuracy": 0,
                    "communication": 0,
                    "confidence": "Low",
                    "recommendation": "Not enough responses to evaluate the candidate."
                }
            }
            return jsonify({"status": "success", "report": fallback_report})

        # Proceed with LLM generation if conversation is valid
        user_prompt = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in filtered_messages]
        )
        full_prompt = f"""Generate a concise and professional interview feedback report based on this conversation:\n\n{user_prompt}"""

        client = Groq(api_key=os.getenv("GROQ_REPORT_API_KEY"))
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": INTERVIEW_PROFILES['report']['instructions']},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )

        report_text = completion.choices[0].message.content.strip()
        with open("llm_output_log.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

        # Extract scores
        score_match = re.search(r"### SCORES ###\s*({.*?})\s*### END SCORES ###", report_text, re.DOTALL)
        if score_match:
            score_json_str = score_match.group(1)
            report_score = json.loads(score_json_str)
        else:
            report_score = {
                "overall_score": 0,
                "accuracy": 0,
                "communication": 0,
                "confidence": "Low",
                "recommendation": "Could not parse scores"
            }
        confidence_map = {"Low": 30, "Medium": 70, "High": 90}
        confidence_score = confidence_map.get(report_score["confidence"], 0)

        report_score["overall_score"] = (report_score["accuracy"] + report_score["communication"] + confidence_score) // 3

        session['latest_report'] = {
            "report_text": report_text,
            "report_score": report_score
        }
        # print(report_score)

        return jsonify({"status": "success", "report": report_text})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template("error.html", message=str(e)), 500


@app.route("/report")
def report():
    latest_report = session.get("latest_report", {})
    report_text = latest_report.get("report_text", "")
    full_convo = [
        msg for msg in session.get("conversation", [])
        if msg.get("role") != "system"
    ]
    def extract_block(text, start_tag, end_tag):
        try:
            return text.split(start_tag)[1].split(end_tag)[0].strip()
        except (IndexError, AttributeError):
            return "Not Available"

    analysis = extract_block(report_text, "### ANALYSIS ###", "### END ANALYSIS ###")
    strengths = extract_block(report_text, "### STRENGTHS ###", "### END STRENGTHS ###").splitlines()
    improvements = extract_block(report_text, "### IMPROVEMENTS ###", "### END IMPROVEMENTS ###").splitlines()

    report_score = latest_report.get("report_score", {
        "overall_score": 0,
        "accuracy": 0,
        "communication": 0,
        "confidence": "Low",
        "recommendation": "Not Available"
    })

    return render_template("report.html",
                           analysis=analysis,
                           strengths=strengths,
                           improvements=improvements,
                           report_score=report_score,
                           conversation=full_convo)


def extract_block(text, start_tag, end_tag):
    try:
        return text.split(start_tag)[1].split(end_tag)[0].strip()
    except (IndexError, AttributeError):
        return "Not Available"

def calculate_overall_score(accuracy, confidence):
    """
    Calculate overall score based only on accuracy and confidence.
    - Accuracy: weighted 70%
    - Confidence: mapped and weighted 30%
    """
    confidence_map = {"Low": 30, "Medium": 70, "High": 90}
    confidence_score = confidence_map.get(confidence, 0)
    print(str(confidence_score)+"\n\n\n\n")
    # Weighted formula
    overall = int((accuracy * 0.7) + (confidence_score * 0.3))
    return overall, confidence_score

@app.route("/dashboard")
def dashboard():
    latest_report = session.get("latest_report", {})
    report_score = latest_report.get("report_score", {})

    # Extract values
    accuracy = report_score.get("accuracy", 0)
    confidence = report_score.get("confidence", "Low")
    communication_score = report_score.get("communication", 0) 

    # Calculate overall score dynamically (accuracy + confidence only)
    overall_score, confidence_score_num = calculate_overall_score(accuracy, confidence)

    # Get response times
    response_times = session.get("response_times", [])

    # Candidate info
    candidate = {
        "name": session.get("candidate_name", "Candidate"),
        "score": overall_score,
        "skills": [accuracy, communication_score, confidence_score_num],
        "response_times": response_times
    }

    # Dynamic labels
    response_time_labels = [f"Q{i+1}" for i in range(len(response_times))]
    skill_labels = ["Accuracy", "Communication", "Confidence"]

    # Extract summary
    report_text = latest_report.get("report_text", "")
    summary = extract_block(report_text, "### ANALYSIS ###", "### END ANALYSIS ###")

    return render_template(
        "dashboard.html",
        candidate=candidate,
        skill_labels=skill_labels,
        response_time_labels=response_time_labels,
        summary=summary
    )

@app.route("/dashboard-data")
def dashboard_data():
    latest_report = session.get("latest_report", {})
    report_score = latest_report.get("report_score", {})

    # Extract values
    accuracy = report_score.get("accuracy", 0)
    confidence = report_score.get("confidence", "Low")
    communication_score = report_score.get("communication", 0)

    # Calculate overall score dynamically
    overall_score, confidence_score_num = calculate_overall_score(accuracy, confidence)

    # Get response times
    response_times = session.get("response_times", [])

    # Candidate info
    candidate = {
        "name": session.get("candidate_name", "Candidate"),
        "score": overall_score,
        "skills": [accuracy, communication_score, confidence_score_num],
        "response_times": response_times
    }

    # Dynamic labels
    response_time_labels = [f"Q{i+1}" for i in range(len(response_times))]
    skill_labels = ["Accuracy", "Communication", "Confidence"]

    return jsonify({
        "candidate": candidate,
        "skill_labels": skill_labels,
        "response_time_labels": response_time_labels
    })

if __name__ == '__main__':
    app.run(
    host='0.0.0.0',
    port=5000,
    debug=False
)