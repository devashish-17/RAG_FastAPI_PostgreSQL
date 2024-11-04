from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import os

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from app import auth, models, schemas, security
from app.db import get_db
from app.models import User
from app.prompts import generate_context, promptTemplate, getPromptsAndReturnResponse, createChunks, uploadAndSplitPdfFile

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

router = APIRouter()

@router.get('/get_loggedIn_user')
async def get_loggedIn_user(current_user: schemas.UserInDB = Depends(auth.get_current_user)):
    return {
        "conversation": "This is a secure conversation",
        "current_user": current_user.username
    }

@router.post("/register/", response_model=schemas.UserInDBBase)
async def register(user_in: schemas.UserIn, db: Session = Depends(get_db)):
    db_user = auth.get_user(db, username=user_in.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    db_user = db.query(models.User).filter(models.User.email == user_in.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = security.get_password_hash(user_in.password)
    db_user = models.User(
        **user_in.dict(exclude={"password"}), hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@router.post("/token", response_model=schemas.Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = auth.get_user(db, username=form_data.username)
    if not user or not security.pwd_context.verify(
        form_data.password, user.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}



# Load the PDF and create chunks once at the start
loader = uploadAndSplitPdfFile('Evolution_of_AI.pdf')
pages = createChunks(loader, 500, 100)
retrieval_chain = promptTemplate("You are a helpful AI assistant. Answer based on the context provided. If you don't know, say I don't know context: {context} input: {input} answer:", pages)

@router.post("/conversation/")
async def read_conversation(
    query: str,
    current_user: schemas.UserInDB = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    db_user = db.query(User).get(current_user.id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prepare context from user information
    context = generate_context(db_user)

    # Get response from the retrieval chain
    response = getPromptsAndReturnResponse(query, context, retrieval_chain)

    return {"response": response}
