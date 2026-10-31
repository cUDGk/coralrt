# Paths for the WSL scripts (sourced). REPO: coralrt on the Windows side; WORK: the folder around it, which holds
# test_data. GIT: the git repository coralrt is committed in (coralrt itself, or a folder above it); build.sh clones
# it into ~/coral, and LIN is coralrt in that copy.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORK=$(dirname "$REPO")
if [ -d "$REPO/.git" ]; then
  GIT=$REPO LIN=$HOME/coral
else
  GIT=$WORK LIN=$HOME/coral/$(basename "$REPO")
fi
