# Public Safety First Response System

A prototype public-safety first-response system that links **smart CCTV-based early incident recognition** with **safety-aware drone response** in urban environments.

This project treats the drone not as an isolated vehicle, but as part of a broader **public-safety workflow**:

**early recognition -> dispatch recommendation -> human approval -> drone first response**

---

## Overview

The system is organized into three layers:

- **Perception layer**: smart CCTV or citizen-report-based early incident recognition
- **Decision-support layer**: temporal verification and human-in-the-loop dispatch recommendation
- **Drone response layer**: risk-aware navigation, anomaly detection, and emergency mitigation

The main design principle is:

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
    G --> H["Follow-up response"]
