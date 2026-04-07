try:
    from easy import run_task as run_easy, TASK_NAME as EASY
    from medium import run_task as run_medium, TASK_NAME as MEDIUM
    from hard import run_task as run_hard, TASK_NAME as HARD
except ImportError:
    from .easy import run_task as run_easy, TASK_NAME as EASY
    from .medium import run_task as run_medium, TASK_NAME as MEDIUM
    from .hard import run_task as run_hard, TASK_NAME as HARD

__all__ = ["run_easy", "run_medium", "run_hard", "EASY", "MEDIUM", "HARD"]
