# --- FORCE IPv4 PATCH (CRITICAL FOR RENDER STABILITY) ---
import socket
_original_getaddrinfo = socket.getaddrinfo
def getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = getaddrinfo_ipv4
# --------------------------------------------------------

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
from pydub import AudioSegment, effects  # Required for Audio Normalization
import io
import tempfile
import os
import shutil
from typing import List, Dict, Any, Optional
import random
import hashlib
from datetime import datetime, timedelta
import jwt
import json
import base64
import requests 
from dotenv import load_dotenv
from gtts import gTTS
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import re  # For regex cleaning

# Load environment variables
load_dotenv()

app = FastAPI(title="Skillspeak API", version="2.3.0")

# =================================================================
# 🔑  CONFIGURATION
# =================================================================
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
SENDER_EMAIL = "gianangelomendoza@gmail.com"
SECRET_KEY = os.getenv("SECRET_KEY", "secret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
LANGUAGE_CODES = {"English": "en", "Tagalog": "tl", "Spanish": "es", "French": "fr", "Japanese": "ja"}

# =================================================================
# 🧠 SMART ANALYSIS ENGINE (LOGIC & HEURISTICS)
# =================================================================

FILLER_WORDS = {
    "English": ["um", "uh", "like", "you know", "actually", "basically", "literally", "i mean", "sort of"],
    "Tagalog": ["ano", "parang", "kuwan", "ah", "eh", "bale", "kumbaga", "siguro", "di ba"]
}

def generate_smart_suggestions(wpm: int, filler_count: int, language: str, duration: float) -> List[str]:
    suggestions = []

    # 1. DURATION CHECK
    if duration < 5.0:
        return ["The recording was too short to analyze accurately. Try speaking for at least 10 seconds."]

    # 2. PACE ANALYSIS (WPM)
    if wpm < 90:
        suggestions.append("You are speaking quite slowly (< 90 WPM). Try to increase your energy and pace to keep listeners engaged.")
    elif wpm > 160:
        suggestions.append("You are speaking very fast (> 160 WPM). Slow down slightly to ensure every word is clear and easy to follow.")
    else:
        suggestions.append("Great pacing! You are within the ideal conversational range (90-150 WPM).")

    # 3. FILLER WORD ANALYSIS
    if filler_count == 0:
        suggestions.append("Excellent flow! No filler words were detected. You sound very confident.")
    elif filler_count <= 2:
        suggestions.append("Good clarity. You kept filler words to a minimum.")
    elif filler_count > 5:
        suggestions.append(f"Detected {filler_count} filler words. Try pausing silently to gather your thoughts instead of using fillers.")

    # 4. LANGUAGE SPECIFIC TIPS
    if language == "Tagalog":
        if filler_count > 0:
            suggestions.append("Sa Tagalog, subukang iwasan ang labis na paggamit ng 'ano' at 'parang' habang nag-iisip.")
        if wpm > 160:
            suggestions.append("Masyadong mabilis ang iyong pagsasalita. Subukang bagalan para mas maintindihan.")
    elif language == "English":
        if "like" in str(suggestions) or filler_count > 3:
            suggestions.append("Try to avoid using 'like' as a connector. Use transition words like 'furthermore' or 'additionally'.")

    return suggestions

def analyze_logic(content: bytes, language: str):
    # 1. PRE-PROCESS: Normalize Audio (CRITICAL FOR TINY MODEL)
    # This boosts volume so the tiny model hears quiet words better.
    try:
        raw_audio = AudioSegment.from_file(io.BytesIO(content))
        normalized_audio = effects.normalize(raw_audio)
        
        # Save to temp file for Whisper to read
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            normalized_audio.export(f.name, format="wav")
            path = f.name
            
        duration = len(normalized_audio) / 1000.0
    except Exception as e:
        print(f"Audio Processing Error: {e}")
        return {"error": "Invalid audio file"}

    # 2. TRANSCRIBE (Using 'tiny' model)
    try:
        # beam_size=5 helps 'tiny' be slightly more accurate
        # task="transcribe" PREVENTS auto-translation (e.g. Tagalog -> English)
        segments, info = model.transcribe(
            path, 
            beam_size=5, 
            language=LANGUAGE_CODES.get(language, "en"),
            task="transcribe" 
        )
        transcript = " ".join([s.text for s in segments]).strip()
        print(f"📝 Transcript ({language}): {transcript}")
    except Exception as e:
        print(f"Transcribe Error: {e}")
        transcript = ""
    finally:
        if os.path.exists(path):
            os.unlink(path)

    # 3. CALCULATE METRICS
    # Clean up transcript for counting (remove punctuation)
    clean_text = re.sub(r'[^\w\s]', '', transcript.lower()) 
    words = clean_text.split()
    word_count = len(words)
    
    # Calculate WPM
    wpm = int((word_count / duration) * 60) if duration > 0 else 0
    
    # Count Fillers
    target_fillers = FILLER_WORDS.get(language, FILLER_WORDS["English"])
    found_fillers = []
    filler_count = 0
    
    for word in words:
        if word in target_fillers:
            found_fillers.append(word)
            filler_count += 1
            
    # Calculate Pronunciation/Clarity Score
    # Base score 100, minus points for excessive fillers or extreme speed
    score = 100
    score -= (filler_count * 3) # -3 points per filler
    if wpm < 80 or wpm > 170: 
        score -= 10 # -10 points for bad pacing
    
    # Clamp score between 0 and 100
    pronunciation_score = max(0, min(100, score))

    # 4. GENERATE SUGGESTIONS
    suggestions = generate_smart_suggestions(wpm, filler_count, language, duration)

    return {
        "transcript": transcript,
        "language": language,
        "wpm": wpm,
        "fillerWords": filler_count,
        "fillerWordsList": found_fillers,
        "pronunciationScore": pronunciation_score,
        "suggestions": suggestions,
        "duration": int(duration)
    }

# =================================================================

# --- STARTUP CHECKS ---
@app.on_event("startup")
async def startup_check():
    print("\n" + "="*40)
    print("🚀 STARTING SKILLSPEAK SERVER (SMART LOGIC V2)...")
    
    if not shutil.which("ffmpeg"):
        print("❌ CRITICAL: FFmpeg is missing!")
    else:
        print("✅ FFmpeg found.")
        
    if BREVO_API_KEY:
        print(f"✅ Brevo API Key Loaded.")
    else:
        print("❌ WARNING: BREVO_API_KEY is missing.")

    print("="*40 + "\n")

    print("⏳ Loading Faster-Whisper Model... (tiny)")
    global model
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ Faster-Whisper Model Loaded!")
    except Exception as e:
        print(f"❌ Failed to load Whisper: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./skillspeak.db"
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- MODELS ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    name = Column(String, nullable=False)
    preferred_language = Column(String, default="English")
    total_sessions = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    email_verified = Column(Boolean, default=False)
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")

class UserSession(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    language = Column(String, nullable=False)
    transcript = Column(Text)
    wpm = Column(Integer)
    filler_words = Column(Integer)
    filler_words_list = Column(Text)
    pronunciation_score = Column(Integer)
    suggestions = Column(Text)
    file_path = Column(String)
    duration = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="sessions")

class EmailVerification(Base):
    __tablename__ = "email_verifications"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True, nullable=False)
    otp_code = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    is_used = Column(Boolean, default=False)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- SCHEMAS ---
class UserRegister(BaseModel):
    email: str; password: str; name: str
class UserLogin(BaseModel):
    email: str; password: str
class UserUpdate(BaseModel):
    name: Optional[str] = None; preferred_language: Optional[str] = None; new_password: Optional[str] = None
class AnalysisResponse(BaseModel):
    transcript: str; language: str; wpm: int; fillerWords: int
    fillerWordsList: List[str]; pronunciationScore: int; suggestions: List[str]; duration: int
class EmailVerificationRequest(BaseModel):
    email: str; otp: str
class ResendOTPRequest(BaseModel):
    email: str

# --- HELPERS ---
def hash_password(p: str) -> str: return hashlib.sha256(p.encode()).hexdigest()

def create_token(uid: int) -> str:
    return jwt.encode({'user_id': uid, 'exp': datetime.utcnow()+timedelta(days=30)}, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> int:
    try: 
        return jwt.decode(token.replace('Bearer ', ''), SECRET_KEY, algorithms=[ALGORITHM]).get('user_id')
    except: 
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(authorization: str = Header(None)):
    if not authorization: 
        raise HTTPException(401, "No auth header")
    return verify_token(authorization)

# --- EMAIL SENDER ---
def send_email(to: str, otp: str):
    print(f"\n🔐 VERIFICATION CODE for {to}: {otp}\n")
    if not BREVO_API_KEY: return False
    api_key = BREVO_API_KEY.strip().strip('"').strip("'")
    url = "https://api.brevo.com/v3/smtp/email"
    payload = {
        "sender": {"name": "Skillspeak App", "email": SENDER_EMAIL},
        "to": [{"email": to}],
        "subject": "Skillspeak Verification Code",
        "htmlContent": f"<h1>{otp}</h1>"
    }
    headers = {"accept": "application/json", "content-type": "application/json", "api-key": api_key}
    try:
        requests.post(url, json=payload, headers=headers)
        return True
    except: return False

# --- ENDPOINTS ---
@app.post("/analyze-file", response_model=AnalysisResponse)
async def analyze(language: str = "English", audio: UploadFile = File(...)):
    return analyze_logic(await audio.read(), language)

@app.post("/register")
async def register(d: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email==d.email, User.email_verified==True).first():
        raise HTTPException(400, "Email taken")
    db.query(User).filter(User.email==d.email).delete()
    db.query(EmailVerification).filter(EmailVerification.email==d.email).delete()
    u = User(email=d.email, password_hash=hash_password(d.password), name=d.name)
    db.add(u); db.commit()
    otp = str(random.randint(100000,999999))
    db.add(EmailVerification(email=d.email, otp_code=otp, expires_at=datetime.utcnow()+timedelta(minutes=10)))
    db.commit()
    send_email(d.email, otp)
    return {"success": True, "message": "Registration successful"}

@app.post("/send-verification-email")
async def resend_email_endpoint(req: ResendOTPRequest, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == req.email).first()
    if not u: raise HTTPException(404, "User not found")
    otp = str(random.randint(100000,999999))
    db.add(EmailVerification(email=req.email, otp_code=otp, expires_at=datetime.utcnow()+timedelta(minutes=10)))
    db.commit()
    send_email(req.email, otp)
    return {"success": True, "message": "Email sent"}

@app.post("/login")
async def login(d: UserLogin, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email==d.email, User.password_hash==hash_password(d.password)).first()
    if not u: raise HTTPException(401, "Invalid credentials")
    if not u.email_verified: raise HTTPException(401, "Email not verified")
    return {"token": create_token(u.id), "user_id": u.id, "name": u.name, "email": u.email, "preferred_language": u.preferred_language}

@app.post("/verify-email")
async def verify(d: EmailVerificationRequest, db: Session = Depends(get_db)):
    rec = db.query(EmailVerification).filter(EmailVerification.email==d.email, EmailVerification.otp_code==d.otp, EmailVerification.is_used==False).first()
    if not rec or datetime.utcnow() > rec.expires_at: raise HTTPException(400, "Invalid/Expired OTP")
    rec.is_used = True; u = db.query(User).filter(User.email==d.email).first()
    if u: u.email_verified = True
    db.commit(); return {"success": True}

@app.get("/user/profile")
async def profile(uid: int = Depends(get_current_user), db: Session = Depends(get_db)):
    u = db.query(User).filter(User.id==uid).first()
    return {"id": u.id, "name": u.name, "email": u.email, "preferred_language": u.preferred_language, "total_sessions": u.total_sessions}

@app.get("/text-to-speech")
async def tts(text: str, language: str = "English"):
    try:
        t = gTTS(text=text, lang=LANGUAGE_CODES.get(language, "en"), slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f: t.save(f.name); path=f.name
        with open(path, 'rb') as f: b64 = base64.b64encode(f.read()).decode()
        os.unlink(path); return {"success": True, "audio_base64": b64}
    except Exception as e: return {"success": False, "message": str(e)}

@app.post("/sessions")
async def save_sess(
    language: str = Form(...), transcript: str = Form(...), wpm: int = Form(...),
    filler_words: int = Form(...), pronunciation_score: int = Form(...), duration: int = Form(...),
    file_path: str = Form(""), filler_words_list: str = Form("[]"), suggestions: str = Form("[]"),
    uid: int = Depends(get_current_user), db: Session = Depends(get_db)
):
    s = UserSession(user_id=uid, language=language, transcript=transcript, wpm=wpm, filler_words=filler_words, pronunciation_score=pronunciation_score, duration=duration, file_path=file_path, filler_words_list=filler_words_list, suggestions=suggestions)
    db.add(s); db.query(User).filter(User.id==uid).first().total_sessions += 1; db.commit(); db.refresh(s)
    return {"session_id": s.id}

@app.get("/sessions")
async def get_sess(uid: int = Depends(get_current_user), db: Session = Depends(get_db)):
    sess = db.query(UserSession).filter(UserSession.user_id==uid).order_by(UserSession.created_at.desc()).all()
    return [{"id": s.id, "language": s.language, "transcript": s.transcript, "wpm": s.wpm, "fillerWords": s.filler_words, "fillerWordsList": json.loads(s.filler_words_list), "pronunciationScore": s.pronunciation_score, "suggestions": json.loads(s.suggestions), "filePath": s.file_path, "duration": s.duration, "date": s.created_at} for s in sess]

@app.delete("/sessions/{sid}")
async def del_sess(sid: int, uid: int = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(UserSession).filter(UserSession.id==sid, UserSession.user_id==uid).delete()
    db.query(User).filter(User.id==uid).first().total_sessions -= 1; db.commit()
    return {"message": "Deleted"}

@app.get("/health")
async def health(): return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)