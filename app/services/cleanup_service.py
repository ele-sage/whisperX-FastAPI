import asyncio
import logging
from datetime import datetime, timezone, timedelta

from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import Task
from sqlalchemy import delete

logger = logging.getLogger(__name__)

async def periodic_task_cleanup(days: int = 7) -> None:
    """
    Background loop that runs once every 24 hours to delete old tasks.
    """
    logger.info(f"Started background cleanup service. Old tasks (>{days} days) will be deleted daily.")
    
    while True:
        try:
            # Run the cleanup in a threadpool so it doesn't block the asyncio event loop
            await asyncio.to_thread(_delete_old_tasks, days)
            
            # Sleep for 24 hours (86400 seconds)
            await asyncio.sleep(86400)
            
        except asyncio.CancelledError:
            logger.info("Cleanup service shutting down.")
            break
        except Exception as e:
            logger.error(f"Error in background task cleanup: {e}")
            # Sleep for an hour before retrying on crash
            await asyncio.sleep(3600)

def _delete_old_tasks(days: int) -> None:
    """Synchronous database operation to delete tasks."""
    session = SessionLocal()
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        result = session.execute(
            delete(Task).where(Task.start_time < cutoff_date)
        )
        session.commit()
        deleted_count = result.rowcount
        
        if deleted_count > 0:
            logger.info(f"Database cleanup: Deleted {deleted_count} tasks older than {days} days.")
        else:
            logger.debug(f"Database cleanup: No tasks older than {days} days found.")
            
    except Exception as e:
        session.rollback()
        logger.error(f"Database cleanup failed: {e}")
        raise
    finally:
        session.close()
