from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
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

# Database Imports
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
        print("\n" + "="*50)
        print("❌ CRITICAL ERROR: FFmpeg is missing!")
        print("Audio analysis will fail. Please install FFmpeg.")
        print("Windows: Download from ffmpeg.org and add bin/ to PATH.")
        print("="*50 + "\n")
    else:
        print("✅ FFmpeg found.")

    # 2. Load Whisper Model
    print("⏳ Loading Whisper Model... (This runs once at startup)")
    global model
    try:
        model = whisper.load_model("base")
        print("✅ Whisper Model Loaded!")
    except Exception as e:
        print(f"❌ Failed to load Whisper: {e}")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE SETUP (SQLAlchemy) ---
# This automatically picks Postgres if online, or SQLite if local
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Local fallback
    DATABASE_URL = "sqlite:///./skillspeak.db"
    print("⚠️  No DATABASE_URL found. Using local SQLite database.")
elif DATABASE_URL.startswith("postgres://"):
    # Fix for some cloud providers using legacy postgres://
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create Database Engine
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- DATABASE MODELS ---
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
    filler_words_list = Column(Text) # Stored as JSON string
    pronunciation_score = Column(Integer)
    suggestions = Column(Text)       # Stored as JSON string
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

# Create Tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- CONFIGURATION ---
SECRET_KEY = os.getenv("SECRET_KEY", "skillspeak-fallback-secret-key")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

LANGUAGE_CODES = {
    "English": "en", "Tagalog": "tl", "Spanish": "es", "French": "fr", "Japanese": "ja"
}

# --- Pydantic Schemas ---
class UserRegister(BaseModel):
    email: str
    password: str
    name: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    preferred_language: Optional[str] = None
    new_password: Optional[str] = None

class AnalysisResponse(BaseModel):
    transcript: str
    language: str
    wpm: int
    fillerWords: int
    fillerWordsList: List[str]
    pronunciationScore: int
    suggestions: List[str]
    duration: int

class EmailVerificationRequest(BaseModel):
    email: str
    otp: str

class ResendOTPRequest(BaseModel):
    email: str

# --- HELPER FUNCTIONS ---
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_token(user_id: int) -> str:
    expiration = datetime.utcnow() + timedelta(days=30)
    payload = {'user_id': user_id, 'exp': expiration, 'iat': datetime.utcnow()}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> int:
    try:
        if token.startswith('Bearer '):
            token = token[7:]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get('user_id')
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    return verify_token(authorization)

def send_verification_email(email: str, otp: str):
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("❌ Email credentials missing, skipping email send.")
        return False
    try:
        subject = "Skillspeak - Email Verification"
        body = f"Welcome to Skillspeak!\nYour code is: {otp}"
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

def generate_otp() -> str:
    return str(random.randint(100000, 999999))

# --- ENDPOINTS ---

# 1. Auth
@app.post("/register")
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    # Check existing
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        if existing.email_verified:
            raise HTTPException(status_code=400, detail="Email already registered")
        else:
            # Cleanup unverified
            db.delete(existing)
            db.commit()
    
    new_user = User(
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        name=user_data.name
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    otp = generate_otp()
    new_otp = EmailVerification(
        email=user_data.email, 
        otp_code=otp, 
        expires_at=datetime.utcnow() + timedelta(minutes=10)
    )
    db.add(new_otp)
    db.commit()
    
    send_verification_email(user_data.email, otp)
    return {"success": True, "message": "Registration successful"}

@app.post("/login")
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    pwd_hash = hash_password(user_data.password)
    user = db.query(User).filter(User.email == user_data.email, User.password_hash == pwd_hash).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.email_verified:
        raise HTTPException(status_code=401, detail="Email not verified")
        
    user.last_login = datetime.utcnow()
    db.commit()
    
    token = create_token(user.id)
    return {
        "token": token, "user_id": user.id, "name": user.name,
        "email": user.email, "preferred_language": user.preferred_language
    }

@app.post("/verify-email")
async def verify_email(request: EmailVerificationRequest, db: Session = Depends(get_db)):
    otp_record = db.query(EmailVerification).filter(
        EmailVerification.email == request.email,
        EmailVerification.otp_code == request.otp,
        EmailVerification.is_used == False
    ).order_by(EmailVerification.created_at.desc()).first()
    
    if not otp_record:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    if datetime.utcnow() > otp_record.expires_at:
        raise HTTPException(status_code=400, detail="OTP Expired")
        
    otp_record.is_used = True
    user = db.query(User).filter(User.email == request.email).first()
    if user:
        user.email_verified = True
    db.commit()
    return {"success": True}

@app.get("/user/profile")
async def get_user_profile(user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user.id, "email": user.email, "name": user.name,
        "preferred_language": user.preferred_language, "total_sessions": user.total_sessions,
        "email_verified": user.email_verified
    }

@app.put("/user/profile")
async def update_profile(update_data: UserUpdate, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if update_data.name: user.name = update_data.name
    if update_data.preferred_language: user.preferred_language = update_data.preferred_language
    if update_data.new_password: user.password_hash = hash_password(update_data.new_password)
    
    db.commit()
    return {"message": "Profile updated"}

# 2. TTS
@app.get("/text-to-speech")
async def text_to_speech(text: str, language: str = "English"):
    try:
        lang_code = LANGUAGE_CODES.get(language, "en")
        tts = gTTS(text=text, lang=lang_code, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_path = temp_file.name
        tts.save(temp_path)
        with open(temp_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        os.unlink(temp_path)
        return {"success": True, "audio_base64": b64}
    except Exception as e:
        return {"success": False, "message": str(e)}

# 3. Analysis Logic
FILLER_WORDS = {
    "English": ["um", "uh", "like", "you know", "actually"],
    "Tagalog": ["ano", "alam mo", "parang", "eh", "kuwan"],
    "Spanish": ["eh", "este", "o sea", "bueno"],
    "French": ["euh", "alors", "donc"],
    "Japanese": ["あの", "ええと", "まあ"]
}
SUGGESTIONS = {
    "English": ["Reduce filler words.", "Maintain steady pace.", "Speak clearly."],
    "Tagalog": ["Bawasan ang 'ano' at 'parang'.", "Ayusin ang bilis.", "Lakasan ang boses."],
}

def analyze_logic(content: bytes, language: str):
    # Whisper
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(content)
        path = f.name
    try:
        result = model.transcribe(path)
        transcript = result["text"].strip()
    except Exception as e:
        print(f"Whisper Error: {e}")
        transcript = ""
    finally:
        os.unlink(path)

    # Metrics
    try:
        audio = AudioSegment.from_file(io.BytesIO(content))
        duration = len(audio) / 1000.0
    except:
        duration = 1.0
        
    words = transcript.lower().split()
    wpm = int((len(words) / duration) * 60) if duration > 0 else 0
    
    target_fillers = FILLER_WORDS.get(language, FILLER_WORDS["English"])
    filler_count = sum(transcript.lower().count(f) for f in target_fillers)
    detected = [f for f in target_fillers if f in transcript.lower()]
    
    score = max(0, min(100, 100 - (filler_count * 2)))
    suggs = SUGGESTIONS.get(language, SUGGESTIONS["English"])
    
    return {
        "transcript": transcript, "language": language, "wpm": wpm,
        "fillerWords": filler_count, "fillerWordsList": detected,
        "pronunciationScore": score, "suggestions": suggs, "duration": int(duration)
    }

@app.post("/analyze-file", response_model=AnalysisResponse)
async def analyze_file(language: str = "English", audio: UploadFile = File(...)):
    content = await audio.read()
    return analyze_logic(content, language)

# 4. Sessions
@app.post("/sessions")
async def save_session(
    language: str = Form(...), transcript: str = Form(...), wpm: int = Form(...),
    filler_words: int = Form(...), pronunciation_score: int = Form(...), duration: int = Form(...),
    file_path: str = Form(""), filler_words_list: str = Form("[]"), suggestions: str = Form("[]"),
    user_id: int = Depends(get_current_user), db: Session = Depends(get_db)
):
    # Load raw JSON strings to Python objects to verify, then save as string
    # We store them as strings in DB for simplicity
    new_session = UserSession(
        user_id=user_id, language=language, transcript=transcript, wpm=wpm,
        filler_words=filler_words, pronunciation_score=pronunciation_score,
        duration=duration, file_path=file_path,
        filler_words_list=filler_words_list, suggestions=suggestions
    )
    db.add(new_session)
    
    # Update user stats
    user = db.query(User).filter(User.id == user_id).first()
    if user: user.total_sessions += 1
    
    db.commit()
    db.refresh(new_session)
    return {"session_id": new_session.id}

@app.get("/sessions")
async def get_sessions(user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(UserSession).filter(UserSession.user_id == user_id).order_by(UserSession.created_at.desc()).all()
    result = []
    for s in sessions:
        try: fw_list = json.loads(s.filler_words_list)
        except: fw_list = []
        try: sugg = json.loads(s.suggestions)
        except: sugg = []
        
        result.append({
            "id": s.id, "language": s.language, "transcript": s.transcript,
            "wpm": s.wpm, "fillerWords": s.filler_words,
            "fillerWordsList": fw_list, "pronunciationScore": s.pronunciation_score,
            "suggestions": sugg, "filePath": s.file_path,
            "duration": s.duration, "date": s.created_at
        })
    return result

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: int, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(UserSession).filter(UserSession.id == session_id, UserSession.user_id == user_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db.delete(session)
    user = db.query(User).filter(User.id == user_id).first()
    if user: user.total_sessions = max(0, user.total_sessions - 1)
    
    db.commit()
    return {"message": "Deleted"}

# Admin
@app.get("/admin/users")
async def admin_users(user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [{"id": u.id, "email": u.email, "name": u.name, "total_sessions": u.total_sessions} for u in users]

@app.get("/admin/stats")
async def admin_stats(user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    total_users = db.query(User).count()
    total_sessions = db.query(UserSession).count()
    return {"total_users": total_users, "total_sessions": total_sessions}

@app.get("/health")
async def health():
    return {"status": "healthy", "database": "active"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)