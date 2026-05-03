# Public Safety First Response System

A prototype public-safety first-response system that links **smart CCTV-based early incident recognition** with **safety-aware drone response** in urban environments.

This project is not just a drone project. Its top-level goal is to support **early public-safety response** through the following workflow:

1. Smart CCTV or a citizen report detects an early risk signal
2. AI analyzes the signal and produces a dispatch recommendation
3. A human operator makes the final launch decision
4. A drone performs first response while minimizing secondary risk

---

## Overview

The system is organized into three layers:

- **Perception layer**: smart CCTV-based early risk recognition
- **Decision-support layer**: temporal verification and human-in-the-loop dispatch recommendation
- **Drone response layer**: safe route planning, anomaly detection, and emergency mitigation

The key design principle of the drone stack is **third-party harm minimization**, not aircraft preservation.  
In other words, the system prioritizes:

**human safety > infrastructure safety > vehicle safety > payload**

---

## System Flow

```mermaid
flowchart LR
    A["Risk event occurs"] --> B["Early recognition<br/>Smart CCTV or citizen report"]
    B --> C["Situation assessment<br/>type, location, urgency"]
    C --> D["Dispatch recommendation<br/>AI decision support"]
    D --> E["Human approval"]
    E --> F["Drone first response mission"]
    F --> G["On-site observation / information relay"]
    G --> H["Follow-up response<br/>police, fire department, control center"]
