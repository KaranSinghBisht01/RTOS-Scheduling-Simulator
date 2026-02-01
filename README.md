# RTOS Scheduling Simulator (Web-Based)

## 📌 Project Overview
This project is a **Web-Based RTOS Scheduling Simulator** developed using **Python and Flask**.  
It simulates how an operating system schedules processes using different **CPU scheduling algorithms**, tracks process state transitions, and calculates performance metrics.

The system allows users to manually define processes, select a scheduling algorithm, and observe execution timelines, system performance, and simulated RTOS monitoring data.

> ⚠️ Note: This project is a **simulation for educational purposes**, not a real RTOS kernel.

---

## 🎯 Objectives
- Demonstrate how RTOS scheduling algorithms work
- Visualize process execution and state transitions
- Analyze scheduling performance metrics
- Understand RTOS concepts such as context switching and IPC

---

## ⚙️ Key Features
- Web-based interface using Flask
- Manual process creation (arrival time, burst time, priority, I/O)
- Algorithm selection at runtime
- Process state tracking (NEW, READY, RUNNING, WAITING, TERMINATED)
- Execution timeline (Gantt-style data)
- Performance analysis
- Simulated system & IPC monitoring

---

## 🧠 Scheduling Algorithms Implemented
- First Come First Serve (FCFS)
- Shortest Job First (SJF – Non-preemptive)
- Shortest Remaining Time First (SRTF – Preemptive)
- Priority Scheduling (Non-preemptive)
- Round Robin (Configurable Time Quantum)

---

## 🔄 Process States
Each process transitions through the following states:
- NEW
- READY
- RUNNING
- WAITING (I/O)
- TERMINATED

State transitions are logged with timestamps.

---

## 📊 Performance Metrics
For each simulation:
- Waiting Time
- Turnaround Time
- Response Time
- CPU Utilization
- Throughput
- Context Switch Count

---

## 🖥️ System Monitoring (Simulated)
The system provides simulated RTOS monitoring data:
- CPU load & utilization
- Memory usage (heap & stack)
- Context switch rate
- IPC mechanisms:
  - Semaphores
  - Mutexes
  - Message queues

---

## 📥 Input Details
User can provide:
- Process ID
- Arrival Time
- Burst Time
- Priority (for priority scheduling)
- I/O request timing & duration
- Time Quantum (for Round Robin)

Inputs are accepted via API or UI.

---

## ▶️ How to Run the Project

### Prerequisites
- Python 3.x
- Flask

### Steps
1. Clone the repository
2. Navigate to the project folder
3. Install dependencies:
   ```bash
   pip install flask
