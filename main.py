from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
from pydub import AudioSegment
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from gtts import gTTS
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

# Load environment variables
load_dotenv()

app = FastAPI(title="Skillspeak API", version="2.0.0")

# --- STARTUP CHECKS ---
@app.on_event("startup")
async def startup_check():
    # 1. Check for FFmpeg
    if not shutil.which("ffmpeg"):
        print("❌ CRITICAL ERROR: FFmpeg is missing!")
    else:
        print("✅ FFmpeg found.")

    # 2. Load Faster-Whisper Model
    print("⏳ Loading Faster-Whisper Model... (tiny)")
    global model
    try:
        # 'tiny' uses very little RAM (~70MB). 
        # 'cpu' is required for Render Free Tier (no GPU).
        # 'int8' quantization makes it even smaller.
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
    print("⚠️  No DATABASE_URL found. Using local SQLite.")
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

# --- CONFIG ---
SECRET_KEY = os.getenv("SECRET_KEY", "secret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
LANGUAGE_CODES = {"English": "en", "Tagalog": "tl", "Spanish": "es", "French": "fr", "Japanese": "ja"}

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

# --- HELPERS ---
def hash_password(p: str) -> str: return hashlib.sha256(p.encode()).hexdigest()
def create_token(uid: int) -> str:
    return jwt.encode({'user_id': uid, 'exp': datetime.utcnow()+timedelta(days=30)}, SECRET_KEY, algorithm=ALGORITHM)
def verify_token(token: str) -> int:
    try: return jwt.decode(token.replace('Bearer ', ''), SECRET_KEY, algorithms=[ALGORITHM]).get('user_id')
    except: raise HTTPException(status_code=401, detail="Invalid token")
async def get_current_user(auth: str = Header(None)):
    if not auth: raise HTTPException(401, "No auth header"); return verify_token(auth)
def send_email(to: str, otp: str):
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD: return False
    try:
        s = smtplib.SMTP(SMTP_SERVER, SMTP_PORT); s.starttls()
        s.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        msg = MIMEMultipart(); msg['From']=EMAIL_ADDRESS; msg['To']=to; msg['Subject']="Skillspeak Code"
        msg.attach(MIMEText(f"Your code: {otp}", 'plain'))
        s.sendmail(EMAIL_ADDRESS, to, msg.as_string()); s.quit(); return True
    except Exception as e: print(e); return False

# --- ENDPOINTS ---
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
    db.commit(); send_email(d.email, otp)
    return {"success": True}

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

# ANALYSIS (Updated for Faster-Whisper)
FILLER_WORDS = {"English": ["um", "uh", "like"], "Tagalog": ["ano", "parang"]}
SUGGESTIONS = {"English": ["Reduce fillers"], "Tagalog": ["Bawasan ang fillers"]}

def analyze_logic(content: bytes, language: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f: f.write(content); path=f.name
    try:
        # Faster-Whisper Transcribe
        segments, info = model.transcribe(path, beam_size=1)
        transcript = " ".join([s.text for s in segments]).strip()
    except Exception as e: print(e); transcript = ""
    finally: os.unlink(path)

    try: duration = len(AudioSegment.from_file(io.BytesIO(content))) / 1000.0
    except: duration = 1.0
    
    words = transcript.lower().split()
    wpm = int((len(words)/duration)*60) if duration>0 else 0
    fillers = [f for f in FILLER_WORDS.get(language, FILLER_WORDS["English"]) if f in transcript.lower()]
    count = sum(transcript.lower().count(f) for f in FILLER_WORDS.get(language, FILLER_WORDS["English"]))
    
    return {
        "transcript": transcript, "language": language, "wpm": wpm, "fillerWords": count,
        "fillerWordsList": fillers, "pronunciationScore": max(0, 100-count*2),
        "suggestions": SUGGESTIONS.get(language, ["Practice more"]), "duration": int(duration)
    }

@app.post("/analyze-file", response_model=AnalysisResponse)
async def analyze(language: str = "English", audio: UploadFile = File(...)):
    return analyze_logic(await audio.read(), language)

# SESSIONS
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