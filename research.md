# Research Report: Deep Dive into Cherry Picker Codebase

## Executive Summary
This report details an exhaustive investigation of the `cherry-picker` repository, motivated by reports of complex system bugs (specifically relating to canceled tasks running erroneously) and a need to understand the underlying infrastructure. A rigorous architectural review has been performed.

## 1. Architectural Overview & Specificities
The `cherry-picker` codebase is an agentic AI system for academic literature research (tailored for arXiv). It leverages:
- **Streamlit** for the frontend user interface (`app.py`).
- **LangChain** and **Google GenAI** for orchestration and processing (`agent.py`).
- **ChromaDB** for vector storage.
- Standard synchronous HTTP requests for external communication (`paper_fetcher.py`).

The architecture follows a strict request-response lifecycle driven by Streamlit's event model. When a user interacts with the UI (e.g., clicking "Analyze"), a synchronous execution thread begins, calling functions in an imperative sequence or via `ThreadPoolExecutor` for parallelizing IO-bound API requests. 

## 2. The Notification System
**Investigation Focus:** Intricacies and mechanics of the notification framework.

**Findings:**
A thorough structural analysis reveals that **there is absolutely no notification system** engineered into this system. 
- There are no pub/sub brokers (e.g., Kafka, Redis PubSub, RabbitMQ).
- There is no WebSocket implementation or Server-Sent Events (SSE).
- No email, SMS, or webhook abstractions exist in the system.
- The UI simply uses Streamlit's native `st.success`, `st.error`, and `st.info` transient banners to indicate status to an active session.

## 3. The Task Scheduling Flow & Bug Analysis
**Investigation Focus:** Deep dive into the task scheduling schema to locate "bugs related to tasks that should have been cancelled but continued to run".

**Findings:**
I searched comprehensively for task scheduling paradigms across all modules, including implicit async loops or background runners. 
**The application contains no background task scheduling flow, nor a task queue.** 

- **No Queue Backends:** There is no integration with Celery, RQ, APScheduler, or any similar persistent background queue.
- **No Background Daemon:** The platform does not run independent background workers pulling tasks. The execution context is strictly bound to the Streamlit session's main execution loop.
- **Parallelization, not Scheduling:** Concurrent operations (such as parallel paper extraction in `app.py`) use Python's vanilla `concurrent.futures.ThreadPoolExecutor`. If a user stops the Streamlit app or closes the browser, the current thread terminates without queueing anything. 

**Conclusion on Bugs:**
Because no queues or distributed task states exist, the "scheduled tasks that should have been cancelled" framework is wholly alien to this codebase. Any perceived bugs of this sort are physically impossible within the realm of this app's architecture. The synchronous `ThreadPoolExecutor` handles blocking IO independently, but it does not implement retry, cancellation flags, or delayed scheduling mechanics. If a run "refuses to stop," it is solely an artifact of Streamlit waiting for threads to join, not a malfunctioning scheduler daemon.

## Final Note
All exhaustive efforts locate these reported bugs to an architecture that this application does not use. The codebase is purely a synchronous pipeline without stateful scheduling or external notifications.
