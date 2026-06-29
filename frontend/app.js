const API_BASE = "http://localhost:8000";

let gameId = null;
let board = [];
let boardSize = 15;
let currentPlayer = 1;
let winner = null;
let lastMove = null;
let humanPlayer = 1;
let difficulty = "medium";

const statusEl = document.getElementById("status");
const boardEl = document.getElementById("board");
const newGameBtn = document.getElementById("new-game");
const difficultySelect = document.getElementById("difficulty");
const playerSelect = document.getElementById("player");

function setStatus(message) {
  statusEl.textContent = message;
}

function getCellLabel(value) {
  if (value === 1) return "X";
  if (value === -1) return "O";
  return "";
}

function isHumanTurn() {
  return winner === null && currentPlayer === humanPlayer;
}

function renderBoard() {
  boardEl.style.gridTemplateColumns = `repeat(${boardSize}, var(--cell-size))`;
  boardEl.innerHTML = "";

  board.forEach((row, r) => {
    row.forEach((cell, c) => {
      const div = document.createElement("div");
      div.className = "cell";
      if (cell === 1) div.classList.add("x");
      if (cell === -1) div.classList.add("o");
      if (lastMove && lastMove[0] === r && lastMove[1] === c) {
        div.classList.add("last-move");
      }
      div.textContent = getCellLabel(cell);
      if (cell === 0 && isHumanTurn()) {
        div.classList.add("clickable");
        div.addEventListener("click", () => makeMove(r, c));
      }
      boardEl.appendChild(div);
    });
  });
}

function updateStatus() {
  if (winner === 1) {
    setStatus("X wins!");
  } else if (winner === -1) {
    setStatus("O wins!");
  } else if (winner === 0) {
    setStatus("Draw.");
  } else if (isHumanTurn()) {
    setStatus("Your turn.");
  } else {
    setStatus("AI is thinking...");
  }
}

function applyState(data) {
  gameId = data.game_id;
  board = data.board;
  boardSize = board.length;
  currentPlayer = data.current_player;
  winner = data.winner;
  lastMove = data.last_move;
  renderBoard();
  updateStatus();
}

async function createGame() {
  setStatus("Starting new game...");
  const response = await fetch(`${API_BASE}/games`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      board_size: boardSize,
      human_player: humanPlayer,
      difficulty,
    }),
  });
  if (!response.ok) {
    const err = await response.json();
    setStatus(`Error: ${err.detail || "Unable to start"}`);
    return;
  }
  const data = await response.json();
  applyState(data);
}

async function makeMove(row, col) {
  if (!gameId || !isHumanTurn()) return;
  setStatus("Submitting move...");
  const response = await fetch(`${API_BASE}/games/${gameId}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ row, col }),
  });
  if (!response.ok) {
    const err = await response.json();
    setStatus(`Error: ${err.detail || "Invalid move"}`);
    return;
  }
  const data = await response.json();
  applyState(data);
}

newGameBtn.addEventListener("click", () => {
  difficulty = difficultySelect.value;
  humanPlayer = parseInt(playerSelect.value, 10);
  createGame();
});

difficultySelect.addEventListener("change", () => {
  difficulty = difficultySelect.value;
});

playerSelect.addEventListener("change", () => {
  humanPlayer = parseInt(playerSelect.value, 10);
});

createGame();
