#!/usr/bin/env bash
#
# ai-risk-upgrade.sh — Automated Git Worktree + Claude Code Sub-Agent Orchestrator
#
# Usage:
#   chmod +x ai-risk-upgrade.sh
#   ./ai-risk-upgrade.sh setup          # Initial setup
#   ./ai-risk-upgrade.sh wave1          # Phase 1 (Foundation)
#   ./ai-risk-upgrade.sh wave2          # Phases 2-5 (Parallel)
#   ./ai-risk-upgrade.sh wave3          # Phases 6-7 (Parallel)
#   ./ai-risk-upgrade.sh wave4          # Phase 8 (Polish)
#   ./ai-risk-upgrade.sh cleanup        # Remove worktrees + branches
#   ./ai-risk-upgrade.sh all            # Run everything sequentially
#
# Git Standards:
#   Branch naming : <type>/<scope>-<short-description>
#   Commit format : <type>(<scope>): <imperative mood description>
#   Types         : feat, fix, refactor, docs, test, chore, ci, perf
#   Scopes        : core, redteam, judge, compliance, guardrails, pipeline, dashboard
#

set -euo pipefail

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
REPO_NAME="ai-risk-evaluation-workbench"
MAIN_BRANCH="main"
UPGRADE_BRANCH="upgrade/2026-compliance-platform"
WORKTREE_DIR="../worktrees"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
log_info()    { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"; }

check_git_clean() {
    if [[ -n $(git status --porcelain) ]]; then
        log_error "Working directory is not clean. Commit or stash changes first."
        git status --short
        exit 1
    fi
}

branch_exists() {
    git show-ref --verify --quiet "refs/heads/$1" 2>/dev/null
}

worktree_exists() {
    git worktree list --porcelain | grep -q "worktree $1" 2>/dev/null
}

# ──────────────────────────────────────────────
# Phase Definitions (Git Standards Compliant)
# ──────────────────────────────────────────────
# Format: BRANCH_NAME | COMMIT_MESSAGE | CLAUDE_PROMPT_FILE

declare -A PHASE_BRANCH=(
    [1]="refactor/core-foundation-architecture"
    [2]="feature/redteam-multi-turn-agent"
    [3]="feature/judge-llm-ensemble"
    [4]="feature/compliance-regulatory-mapping"
    [5]="feature/guardrails-production"
    [6]="ci/cd-eval-pipeline"
    [7]="feature/dashboard-streamlit"
    [8]="docs/polish-readme-packaging"
)

declare -A PHASE_COMMIT=(
    [1]="refactor(core): add Pydantic v2 models, config system, and unified backend interface"
    [2]="feat(redteam): implement multi-turn adversarial agent with 8 attack strategies"
    [3]="feat(judge): add calibrated 3-model LLM-as-Judge ensemble with bias detection"
    [4]="feat(compliance): map eval findings to EU AI Act, NIST RMF, and ISO 42001"
    [5]="feat(guardrails): integrate Presidio PII, toxicity scoring, and injection detection"
    [6]="ci(pipeline): add GitHub Actions eval workflow with regression detection"
    [7]="feat(dashboard): build 5-page Streamlit dashboard with radar charts and attack trees"
    [8]="docs: add architecture diagram, demo data, and finalize packaging"
)

declare -A PHASE_SCOPE=(
    [1]="core"
    [2]="redteam"
    [3]="judge"
    [4]="compliance"
    [5]="guardrails"
    [6]="pipeline"
    [7]="dashboard"
    [8]="docs"
)

declare -A PHASE_DESC=(
    [1]="Foundation Refactor"
    [2]="Multi-Turn Red-Team Agent"
    [3]="LLM-as-Judge Ensemble"
    [4]="Compliance Mapping"
    [5]="Production Guardrails"
    [6]="CI/CD Pipeline"
    [7]="Streamlit Dashboard"
    [8]="Polish & README"
)

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
cmd_setup() {
    log_step "SETUP: Initializing upgrade workspace"

    check_git_clean

    # Ensure we're on main
    git checkout "$MAIN_BRANCH"
    git pull origin "$MAIN_BRANCH" 2>/dev/null || log_warn "Could not pull from origin. Continuing locally."

    # Create upgrade branch if it doesn't exist
    if ! branch_exists "$UPGRADE_BRANCH"; then
        git checkout -b "$UPGRADE_BRANCH"
        log_success "Created branch: $UPGRADE_BRANCH"
    else
        git checkout "$UPGRADE_BRANCH"
        log_info "Branch $UPGRADE_BRANCH already exists. Checked out."
    fi

    # Create worktree directory
    mkdir -p "$WORKTREE_DIR"
    log_success "Worktree directory: $WORKTREE_DIR"

    # Verify CLAUDE.md and UPGRADE_PLAN.md exist
    if [[ ! -f "CLAUDE.md" ]]; then
        log_error "CLAUDE.md not found in repo root. Create it first."
        exit 1
    fi
    if [[ ! -f "UPGRADE_PLAN.md" ]]; then
        log_error "UPGRADE_PLAN.md not found in repo root. Create it first."
        exit 1
    fi
    log_success "CLAUDE.md and UPGRADE_PLAN.md found."

    log_success "Setup complete. Run: ./ai-risk-upgrade.sh wave1"
}

# ──────────────────────────────────────────────
# Create Worktree
# ──────────────────────────────────────────────
create_worktree() {
    local phase=$1
    local branch="${PHASE_BRANCH[$phase]}"
    local wt_path="${WORKTREE_DIR}/phase-${phase}"

    if worktree_exists "$wt_path"; then
        log_warn "Worktree already exists: $wt_path"
        return 0
    fi

    if branch_exists "$branch"; then
        git worktree add "$wt_path" "$branch"
        log_info "Worktree created (existing branch): $wt_path → $branch"
    else
        git worktree add -b "$branch" "$wt_path" "$UPGRADE_BRANCH"
        log_success "Worktree created (new branch): $wt_path → $branch"
    fi
}

# ──────────────────────────────────────────────
# Run Claude Code in Worktree
# ──────────────────────────────────────────────
run_claude_agent() {
    local phase=$1
    local wt_path="${WORKTREE_DIR}/phase-${phase}"
    local scope="${PHASE_SCOPE[$phase]}"
    local desc="${PHASE_DESC[$phase]}"
    local commit_msg="${PHASE_COMMIT[$phase]}"

    log_step "PHASE $phase: $desc"
    log_info "Worktree : $wt_path"
    log_info "Branch   : ${PHASE_BRANCH[$phase]}"
    log_info "Commit   : $commit_msg"

    # Write the prompt file for Claude Code
    cat > "${wt_path}/.claude-prompt-phase${phase}.md" << PROMPT
# Phase $phase: $desc

## Instructions
Read CLAUDE.md and UPGRADE_PLAN.md in this directory.
Execute Phase $phase: $desc — all tasks listed under it.

## Rules
1. Follow the repo structure in CLAUDE.md exactly.
2. All models use Pydantic v2 with strict validation.
3. Every function must have type hints and docstrings.
4. Write tests for ALL new code in tests/.
5. Run \`pytest tests/ -v\` and ensure ALL tests pass before finishing.
6. Do NOT hardcode API keys — use environment variables.
7. Do NOT modify files outside the src/${scope}/ and tests/ directories.

## Commit
When all tasks are complete and tests pass, commit with EXACTLY this message:

\`\`\`
$commit_msg
\`\`\`

Use conventional commit format. Do NOT add co-authored-by lines.
PROMPT

    log_info "Prompt written to: ${wt_path}/.claude-prompt-phase${phase}.md"
    log_info ""
    log_info "Now open a NEW terminal and run:"
    echo ""
    echo -e "  ${GREEN}cd ${wt_path}${NC}"
    echo -e "  ${GREEN}claude${NC}"
    echo ""
    echo -e "  Then paste:"
    echo -e "  ${YELLOW}Read .claude-prompt-phase${phase}.md and execute all instructions.${NC}"
    echo ""
}

# ──────────────────────────────────────────────
# Merge Phase Branch
# ──────────────────────────────────────────────
merge_phase() {
    local phase=$1
    local branch="${PHASE_BRANCH[$phase]}"
    local desc="${PHASE_DESC[$phase]}"

    log_info "Merging: $branch → $UPGRADE_BRANCH"

    git checkout "$UPGRADE_BRANCH"

    if branch_exists "$branch"; then
        git merge --no-ff "$branch" -m "merge: integrate Phase $phase — $desc"
        log_success "Merged: $branch"
    else
        log_error "Branch $branch does not exist. Cannot merge."
        exit 1
    fi
}

# ──────────────────────────────────────────────
# Wave Commands
# ──────────────────────────────────────────────
cmd_wave1() {
    log_step "WAVE 1: Foundation (Single Agent)"
    check_git_clean
    create_worktree 1
    run_claude_agent 1

    echo ""
    log_warn "After Phase 1 agent finishes, come back and run:"
    echo -e "  ${GREEN}./ai-risk-upgrade.sh merge1${NC}"
}

cmd_merge1() {
    merge_phase 1
    log_success "Wave 1 complete. Run: ./ai-risk-upgrade.sh wave2"
}

cmd_wave2() {
    log_step "WAVE 2: Parallel Agents (Phases 2, 3, 4, 5)"
    check_git_clean

    for phase in 2 3 4 5; do
        create_worktree "$phase"
        run_claude_agent "$phase"
        echo ""
    done

    log_warn "Open 4 terminals and run each agent."
    log_warn "After ALL 4 finish, run:"
    echo -e "  ${GREEN}./ai-risk-upgrade.sh merge2${NC}"
}

cmd_merge2() {
    for phase in 2 3 4 5; do
        merge_phase "$phase"
    done
    log_success "Wave 2 complete. Run: ./ai-risk-upgrade.sh wave3"
}

cmd_wave3() {
    log_step "WAVE 3: Parallel Agents (Phases 6, 7)"
    check_git_clean

    for phase in 6 7; do
        create_worktree "$phase"
        run_claude_agent "$phase"
        echo ""
    done

    log_warn "Open 2 terminals and run each agent."
    log_warn "After BOTH finish, run:"
    echo -e "  ${GREEN}./ai-risk-upgrade.sh merge3${NC}"
}

cmd_merge3() {
    for phase in 6 7; do
        merge_phase "$phase"
    done
    log_success "Wave 3 complete. Run: ./ai-risk-upgrade.sh wave4"
}

cmd_wave4() {
    log_step "WAVE 4: Final Polish (Single Agent)"
    check_git_clean
    create_worktree 8
    run_claude_agent 8

    echo ""
    log_warn "After Phase 8 agent finishes, run:"
    echo -e "  ${GREEN}./ai-risk-upgrade.sh merge4${NC}"
}

cmd_merge4() {
    merge_phase 8
    log_success "Wave 4 complete. Run: ./ai-risk-upgrade.sh cleanup"
}

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────
cmd_cleanup() {
    log_step "CLEANUP: Removing worktrees and branches"

    for phase in 1 2 3 4 5 6 7 8; do
        local wt_path="${WORKTREE_DIR}/phase-${phase}"
        local branch="${PHASE_BRANCH[$phase]}"

        if worktree_exists "$wt_path"; then
            git worktree remove --force "$wt_path" 2>/dev/null || true
            log_info "Removed worktree: $wt_path"
        fi

        if branch_exists "$branch"; then
            git branch -d "$branch" 2>/dev/null || git branch -D "$branch"
            log_info "Deleted branch: $branch"
        fi
    done

    # Remove worktree directory if empty
    rmdir "$WORKTREE_DIR" 2>/dev/null || true

    # Remove prompt files
    rm -f .claude-prompt-phase*.md 2>/dev/null || true

    log_success "Cleanup complete."
    log_info "Final branch: $UPGRADE_BRANCH"
    log_info "Push with: git push origin $UPGRADE_BRANCH"
}

# ──────────────────────────────────────────────
# Run All (Sequential)
# ──────────────────────────────────────────────
cmd_all() {
    cmd_setup
    cmd_wave1
    echo ""
    log_warn "Run Phase 1 agent, then: ./ai-risk-upgrade.sh merge1"
    log_warn "Then: ./ai-risk-upgrade.sh wave2"
    log_warn "Then: ./ai-risk-upgrade.sh merge2"
    log_warn "Then: ./ai-risk-upgrade.sh wave3"
    log_warn "Then: ./ai-risk-upgrade.sh merge3"
    log_warn "Then: ./ai-risk-upgrade.sh wave4"
    log_warn "Then: ./ai-risk-upgrade.sh merge4"
    log_warn "Then: ./ai-risk-upgrade.sh cleanup"
}

# ──────────────────────────────────────────────
# Status
# ──────────────────────────────────────────────
cmd_status() {
    log_step "STATUS"

    echo -e "${BLUE}Current branch:${NC} $(git branch --show-current)"
    echo ""

    echo -e "${BLUE}Worktrees:${NC}"
    git worktree list
    echo ""

    echo -e "${BLUE}Phase branches:${NC}"
    for phase in 1 2 3 4 5 6 7 8; do
        local branch="${PHASE_BRANCH[$phase]}"
        if branch_exists "$branch"; then
            local last_commit=$(git log -1 --format="%s" "$branch" 2>/dev/null || echo "no commits")
            echo -e "  ${GREEN}✓${NC} Phase $phase: $branch"
            echo -e "    └─ $last_commit"
        else
            echo -e "  ${RED}✗${NC} Phase $phase: $branch (not created)"
        fi
    done
}

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
case "${1:-help}" in
    setup)   cmd_setup ;;
    wave1)   cmd_wave1 ;;
    merge1)  cmd_merge1 ;;
    wave2)   cmd_wave2 ;;
    merge2)  cmd_merge2 ;;
    wave3)   cmd_wave3 ;;
    merge3)  cmd_merge3 ;;
    wave4)   cmd_wave4 ;;
    merge4)  cmd_merge4 ;;
    cleanup) cmd_cleanup ;;
    status)  cmd_status ;;
    all)     cmd_all ;;
    help|*)
        echo ""
        echo "Usage: ./ai-risk-upgrade.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup     Initialize upgrade branch and workspace"
        echo "  wave1     Create worktree for Phase 1 (Foundation)"
        echo "  merge1    Merge Phase 1 after agent completes"
        echo "  wave2     Create worktrees for Phases 2-5 (Parallel)"
        echo "  merge2    Merge Phases 2-5 after agents complete"
        echo "  wave3     Create worktrees for Phases 6-7 (Parallel)"
        echo "  merge3    Merge Phases 6-7 after agents complete"
        echo "  wave4     Create worktree for Phase 8 (Polish)"
        echo "  merge4    Merge Phase 8 after agent completes"
        echo "  cleanup   Remove all worktrees and phase branches"
        echo "  status    Show current state of all phases"
        echo "  all       Show full execution order"
        echo ""
        echo "Git Standards:"
        echo "  Branches : <type>/<scope>-<description>"
        echo "  Commits  : <type>(<scope>): <imperative description>"
        echo "  Types    : feat, fix, refactor, docs, test, chore, ci, perf"
        echo ""
        ;;
esac
