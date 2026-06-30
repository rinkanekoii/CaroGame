from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

from pathlib import Path

from fastapi import FastAPI
import uuid
import torch
from typing import Dict, List, Optional
from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.engine import GameSession, load_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(title="CaroNet", version="2.0.0")

# Global state
sessions: Dict[str, GameSession] = {}
# FORCE CPU for testing INT8 speed
device = torch.device("cpu")
model = load_model(str(PROJECT_ROOT / "backend" / "model_latest_int8.onnx"), 15, device)

class GameCreateRequest(BaseModel):
    board_rows: int = 15
    board_cols: int = 15
    human_player: int = -1
    difficulty: str = "medium"
    rule_type: int = 0

class GameStateResponse(BaseModel):
    game_id: str
    board: List[List[int]]
    current_player: int
    winner: Optional[int]
    last_move: Optional[List[int]]
    ai_move: Optional[List[int]]
    top_ai_moves: Optional[List[dict]]
    eval_score: Optional[float] = 0.0

class MoveRequest(BaseModel):
    row: int
    col: int

class CustomGameRequest(BaseModel):
    board: List[List[int]]
    board_rows: int = 15
    board_cols: int = 15
    next_player: int = 1
    difficulty: str = "medium"
    rule_type: int = 0
    human_player: int = 1

@app.post("/games", response_model=GameStateResponse)
def create_game(req: GameCreateRequest) -> dict:
    game_id = str(uuid.uuid4())
    sims_map = {"easy": 10, "medium": 30, "hard": 50, "extreme": 100}
    sims = sims_map.get(req.difficulty, 400)
    
    session = GameSession(
        board_rows=req.board_rows,
        board_cols=req.board_cols,
        human_player=req.human_player,
        sims=sims,
        model=model,
        device=device,
        rule_type=req.rule_type,
    )
    sessions[game_id] = session
    
    ai_move = None
    top_ai_moves = []
    eval_score = 0.0
    if req.human_player == -1:
        move, top_moves, score = session.ai_move()
        if move:
            ai_move = [move[0], move[1]]
            top_ai_moves = top_moves
            eval_score = score
            
    return {
        "game_id": game_id,
        "board": session.board.tolist(),
        "current_player": session.current_player,
        "winner": session.winner,
        "last_move": list(session.last_move) if session.last_move else None,
        "ai_move": ai_move,
        "top_ai_moves": top_ai_moves,
        "eval_score": eval_score
    }

@app.post("/games/custom", response_model=GameStateResponse)
def create_custom_game(req: CustomGameRequest) -> dict:
    game_id = str(uuid.uuid4())
    sims_map = {"easy": 10, "medium": 30, "hard": 50, "extreme": 100}
    sims = sims_map.get(req.difficulty, 400)
    
    session = GameSession(
        board_rows=req.board_rows,
        board_cols=req.board_cols,
        human_player=req.human_player,
        sims=sims,
        model=model,
        device=device,
        rule_type=req.rule_type,
    )
    # Override the blank board with the custom board
    import numpy as np
    session.board = np.array(req.board, dtype=np.int8)
    session.current_player = req.next_player
    session.mcts.reset_root()
    
    sessions[game_id] = session
    
    ai_move = None
    top_ai_moves = []
    eval_score = 0.0
    # If it's AI's turn, let AI move
    if req.next_player != req.human_player:
        move, top_moves, score = session.ai_move()
        if move:
            ai_move = [move[0], move[1]]
            top_ai_moves = top_moves
            eval_score = score
            
    return {
        "game_id": game_id,
        "board": session.board.tolist(),
        "current_player": session.current_player,
        "winner": session.winner,
        "last_move": list(session.last_move) if session.last_move else None,
        "ai_move": ai_move,
        "top_ai_moves": top_ai_moves,
        "eval_score": eval_score
    }

@app.get("/games/{game_id}", response_model=GameStateResponse)
def get_game(game_id: str) -> dict:
    session = sessions.get(game_id)
    if not session:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not Found")
    return {
        "game_id": game_id,
        "board": session.board.tolist(),
        "current_player": session.current_player,
        "winner": session.winner,
        "last_move": list(session.last_move) if session.last_move else None,
        "ai_move": None,
        "top_ai_moves": [],
        "eval_score": 0.0
    }

@app.post("/games/{game_id}/move", response_model=GameStateResponse)
def make_move(game_id: str, req: MoveRequest) -> dict:
    session = sessions.get(game_id)
    if not session:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not Found")
        
    try:
        session.apply_move(req.row, req.col)
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))
        
    ai_move = None
    top_ai_moves = []
    eval_score = 0.0
    if session.winner is None and session.current_player != session.human_player:
        move, top_moves, score = session.ai_move()
        if move:
            ai_move = [move[0], move[1]]
            top_ai_moves = top_moves
            eval_score = score
            
    return {
        "game_id": game_id,
        "board": session.board.tolist(),
        "current_player": session.current_player,
        "winner": session.winner,
        "last_move": list(session.last_move) if session.last_move else None,
        "ai_move": ai_move,
        "top_ai_moves": top_ai_moves,
        "eval_score": eval_score
    }

@app.get("/games/{game_id}/thinking")
def get_thinking_state(game_id: str) -> dict:
    session = sessions.get(game_id)
    if not session:
        return {"top_ai_moves": []}
    return {"top_ai_moves": session.current_mcts_probs}

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
