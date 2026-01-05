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
from pydub import AudioSegment, effects
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
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import re

load_dotenv()

app = FastAPI(title="Skillspeak API", version="3.2.0 (Native Prompts)")

# =================================================================
# 🔑  CONFIGURATION
# =================================================================
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
SENDER_EMAIL = "gianangelomendoza@gmail.com"
SECRET_KEY = os.getenv("SECRET_KEY", "secret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")

LANGUAGE_CODES = {
    "English": "en", 
    "Tagalog": "tl", 
    "Spanish": "es", 
    "French": "fr", 
    "Japanese": "ja"
}

# =================================================================
# 🧠 NATIVE PROMPTS (THE FIX)
# =================================================================
# This primes the "tiny" model with the target language so it stays in character.
NATIVE_PROMPTS = {
    "English": "This is a transcription. It is not a translation.",
    "Tagalog": "Ito ay isang transcription ng Tagalog audio. Huwag i-translate sa English.",
    "Spanish": "Esta es una transcripción exacta del audio en español.",
    "French": "Voici une transcription textuelle de l'enregistrement en français.",
    "Japanese": "これは日本語の音声の書き起こしです。翻訳ではありません。"
}

# =================================================================
# 🧠 SMART ANALYSIS ENGINE
# =================================================================

FILLER_WORDS = {
    "English": ["um", "uh", "like", "you know", "actually", "basically", "literally"],
    "Tagalog": ["ano", "parang", "kuwan", "ah", "eh", "bale", "kumbaga"],
    "Spanish": ["eh", "este", "bueno", "o sea", "pues", "sabes", "entonces"],
    "French": ["euh", "bah", "ben", "genre", "en fait", "tu vois", "du coup"],
    "Japanese": ["eto", "ano", "nanka", "ma", "sono", "eeto"]
}

SUGGESTIONS_DB = {
    "English": {
        "short": "The recording was too short to analyze accurately.",
        "slow": "You are speaking quite slowly (< 90 WPM). Try to increase your energy.",
        "fast": "You are speaking very fast (> 160 WPM). Slow down for better clarity.",
        "perfect": "Great pacing! You are within the ideal conversational range.",
        "no_fillers": "Excellent flow! No filler words were detected.",
        "few_fillers": "Good clarity. You kept filler words to a minimum.",
        "many_fillers": "Detected {count} filler words. Try pausing instead of saying '{word}'.",
    },
    "Tagalog": {
        "short": "Masyadong maikli ang recording para masuri nang maayos.",
        "slow": "Medyo mabagal ang iyong pagsasalita (< 90 WPM). Subukang bilisan nang kaunti.",
        "fast": "Napakabilis mong magsalita (> 160 WPM). Bagalan nang kaunti para mas maintindihan.",
        "perfect": "Ayos ang bilis ng iyong pagsasalita! Nasa tamang bilis ka.",
        "no_fillers": "Ang galing! Walang filler words na narinig.",
        "few_fillers": "Malinaw ang sinabi mo. Kaunti lang ang filler words.",
        "many_fillers": "Nakarinig kami ng {count} na filler words. Subukang huminto sandali sa halip na magsalita ng '{word}'.",
    },
    "Spanish": {
        "short": "La grabación fue demasiado corta para analizarla.",
        "slow": "Estás hablando bastante despacio (< 90 WPM). Intenta aumentar tu energía.",
        "fast": "Estás hablando muy rápido (> 160 WPM). Ve más despacio para mayor claridad.",
        "perfect": "¡Buen ritmo! Estás en el rango ideal de conversación.",
        "no_fillers": "¡Excelente fluidez! No se detectaron muletillas.",
        "few_fillers": "Buena claridad. Mantuviste las muletillas al mínimo.",
        "many_fillers": "Se detectaron {count} muletillas. Intenta hacer una pausa en lugar de decir '{word}'.",
    },
    "French": {
        "short": "L'enregistrement était trop court pour être analysé.",
        "slow": "Vous parlez assez lentement (< 90 WPM). Essayez d'augmenter votre rythme.",
        "fast": "Vous parlez très vite (> 160 WPM). Ralentissez pour plus de clarté.",
        "perfect": "Super rythme ! Vous êtes dans la moyenne idéale.",
        "no_fillers": "Excellent débit ! Aucun mot de remplissage détecté.",
        "few_fillers": "Bonne clarté. Vous avez limité les mots de remplissage.",
        "many_fillers": "{count} mots de remplissage détectés. Essayez de faire une pause au lieu de dire '{word}'.",
    },
    "Japanese": {
        "short": "録音が短すぎて正確に分析できません。",
        "slow": "話すのが少し遅いです (< 90 WPM)。もう少し元気に話してみましょう。",
        "fast": "話すのが速すぎます (> 160 WPM)。もう少しゆっくり話して明瞭にしましょう。",
        "perfect": "素晴らしいペースです！理想的な会話の範囲内です。",
        "no_fillers": "素晴らしい流れです！フィラーワードは検出されませんでした。",
        "few_fillers": "明瞭です。フィラーワードを最小限に抑えています。",
        "many_fillers": "{count} 個のフィラーワードが検出されました。 '{word}' と言う代わりに一呼吸置いてみましょう。",
    }
}

def generate_smart_suggestions(wpm: int, filler_count: int, language: str, duration: float, common_filler: str) -> List[str]:
    lang_key = language if language in SUGGESTIONS_DB else "English"
    templates = SUGGESTIONS_DB[lang_key]
    suggestions = []

    if duration < 5.0:
        return [templates["short"]]

    if wpm < 90:
        suggestions.append(templates["slow"])
    elif wpm > 160:
        suggestions.append(templates["fast"])
    else:
        suggestions.append(templates["perfect"])

    if filler_count == 0:
        suggestions.append(templates["no_fillers"])
    elif filler_count <= 2:
        suggestions.append(templates["few_fillers"])
    elif filler_count > 2:
        msg = templates["many_fillers"].format(count=filler_count, word=common_filler if common_filler else "...")
        suggestions.append(msg)

    return suggestions

def analyze_logic(content: bytes, language: str):
    language = language.capitalize()
    
    # 1. PRE-PROCESS
    try:
        raw_audio = AudioSegment.from_file(io.BytesIO(content))
        normalized_audio = effects.normalize(raw_audio)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            normalized_audio.export(f.name, format="wav")
            path = f.name
            
        duration = len(normalized_audio) / 1000.0
    except Exception as e:
        print(f"Audio Processing Error: {e}")
        return {"error": "Invalid audio file"}

    # 2. TRANSCRIBE (FIXED WITH NATIVE PROMPTS)
    try:
        iso_code = LANGUAGE_CODES.get(language, "en")
        
        # KEY FIX: Use a prompt WRITTEN IN the target language
        native_prompt = NATIVE_PROMPTS.get(language, "This is a transcription.")

        segments, info = model.transcribe(
            path, 
            beam_size=5, 
            language=iso_code,
            task="transcribe",
            initial_prompt=native_prompt  # <--- Priming in the native language
        )
        transcript = " ".join([s.text for s in segments]).strip()
        print(f"📝 Transcript ({language}): {transcript}")
    except Exception as e:
        print(f"Transcribe Error: {e}")
        transcript = ""
    finally:
        if os.path.exists(path):
            os.unlink(path)

    # 3. METRICS
    clean_text = re.sub(r'[^\w\s]', '', transcript.lower()) 
    words = clean_text.split()
    word_count = len(words)
    wpm = int((word_count / duration) * 60) if duration > 0 else 0
    
    target_fillers = FILLER_WORDS.get(language, FILLER_WORDS["English"])
    found_fillers = []
    filler_count = 0
    
    for word in words:
        if word in target_fillers:
            found_fillers.append(word)
            filler_count += 1
            
    most_common_filler = max(set(found_fillers), key=found_fillers.count) if found_fillers else ""

    score = 100
    score -= (filler_count * 3)
    if wpm < 80 or wpm > 170: 
        score -= 10 
    pronunciation_score = max(0, min(100, score))

    suggestions = generate_smart_suggestions(wpm, filler_count, language, duration, most_common_filler)

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

@app.on_event("startup")
async def startup_check():
    print("\n🚀 STARTING SKILLSPEAK SERVER (Tiny + Native Prompts)...")
    if not shutil.which("ffmpeg"): print("❌ CRITICAL: FFmpeg is missing!")
    else: print("✅ FFmpeg found.")
    
    global model
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="int8") 
        print("✅ Faster-Whisper Model Loaded (Tiny)!")
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

class UserRegister(BaseModel):
    email: str; password: str; name: str
class UserLogin(BaseModel):
    email: str; password: str
class AnalysisResponse(BaseModel):
    transcript: str; language: str; wpm: int; fillerWords: int
    fillerWordsList: List[str]; pronunciationScore: int; suggestions: List[str]; duration: int
class EmailVerificationRequest(BaseModel):
    email: str; otp: str
class ResendOTPRequest(BaseModel):
    email: str

def hash_password(p: str) -> str: return hashlib.sha256(p.encode()).hexdigest()
def create_token(uid: int) -> str:
    return jwt.encode({'user_id': uid, 'exp': datetime.utcnow()+timedelta(days=30)}, SECRET_KEY, algorithm=ALGORITHM)
def verify_token(token: str) -> int:
    try: return jwt.decode(token.replace('Bearer ', ''), SECRET_KEY, algorithms=[ALGORITHM]).get('user_id')
    except: raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(authorization: str = Header(None)):
    if not authorization: raise HTTPException(401, "No auth header")
    return verify_token(authorization)

def send_email(to: str, otp: str):
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
    try: requests.post(url, json=payload, headers=headers); return True
    except: return False

@app.post("/analyze-file", response_model=AnalysisResponse)
async def analyze(language: str = "English", audio: UploadFile = File(...)):
    return analyze_logic(await audio.read(), language.capitalize())

@app.post("/register")
async def register(d: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email==d.email, User.email_verified==True).first(): raise HTTPException(400, "Email taken")
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
    u.last_login = datetime.utcnow(); db.commit()
    return {"token": create_token(u.id), "user_id": u.id, "name": u.name, "email": u.email, "preferred_language": u.preferred_language}

@app.post("/sessions")
async def save_sess(
    language: str = Form(...), transcript: str = Form(...), wpm: int = Form(...),
    filler_words: int = Form(...), pronunciation_score: int = Form(...), duration: int = Form(...),
    file_path: str = Form(""), filler_words_list: str = Form("[]"), suggestions: str = Form("[]"),
    uid: int = Depends(get_current_user), db: Session = Depends(get_db)
):
    s = UserSession(user_id=uid, language=language, transcript=transcript, wpm=wpm, filler_words=filler_words, pronunciation_score=pronunciation_score, duration=duration, file_path=file_path, filler_words_list=filler_words_list, suggestions=suggestions)
    db.add(s)
    user = db.query(User).filter(User.id==uid).first()
    if user: user.total_sessions += 1; user.preferred_language = language
    db.commit(); db.refresh(s)
    return {"session_id": s.id}

@app.get("/sessions")
async def get_sess(uid: int = Depends(get_current_user), db: Session = Depends(get_db)):
    sess = db.query(UserSession).filter(UserSession.user_id==uid).order_by(UserSession.created_at.desc()).all()
    return [{"id": s.id, "language": s.language, "transcript": s.transcript, "wpm": s.wpm, "fillerWords": s.filler_words, "fillerWordsList": json.loads(s.filler_words_list), "pronunciationScore": s.pronunciation_score, "suggestions": json.loads(s.suggestions), "filePath": s.file_path, "duration": s.duration, "date": s.created_at} for s in sess]

@app.get("/user/profile")
async def profile(uid: int = Depends(get_current_user), db: Session = Depends(get_db)):
    u = db.query(User).filter(User.id==uid).first()
    return {"id": u.id, "name": u.name, "email": u.email, "preferred_language": u.preferred_language, "total_sessions": u.total_sessions}

@app.get("/health")
async def health(): return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)