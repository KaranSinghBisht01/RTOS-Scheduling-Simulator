from flask import Flask, render_template, request, jsonify
from enum import Enum
from collections import deque
import copy
import random
import time
from typing import Any, Dict, List, Optional

app = Flask(__name__)

class ProcessState(Enum):
    NEW = "New"
    READY = "Ready"
    RUNNING = "Running"
    WAITING = "Waiting"
    TERMINATED = "Terminated"

class Process:
    def __init__(self, pid, arrival, burst, priority=0, io_requests=None):
        self.pid = pid
        self.arrival = arrival
        self.burst = burst
        self.remaining_burst = burst
        self.priority = priority
        self.io_requests = io_requests or []
        self.state = ProcessState.NEW
        self.state_history = []
        self.current_io = None
        self.io_remaining = 0
        
    def add_state_transition(self, new_state, time):
        if self.state != new_state:
            self.state_history.append({
                'from': self.state.value,
                'to': new_state.value,
                'time': time
            })
            self.state = new_state
    
    def has_io_at_time(self, time):
        for io_time, io_duration in self.io_requests:
            if io_time <= time < io_time + io_duration:
                return True, io_duration - (time - io_time)
        return False, 0


class ProcessManager:
    def __init__(self):
        self._processes: List[Dict[str, Any]] = []
        self._last_simulation: Optional[Dict[str, Any]] = None

    def list_processes(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._processes)

    def add_process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "burst" not in payload:
            raise ValueError("Burst time is required")

        burst = int(payload["burst"])
        if burst <= 0:
            raise ValueError("Burst time must be greater than 0")

        pid = payload.get("pid") or f"T{len(self._processes) + 1}"
        if any(proc["pid"] == pid for proc in self._processes):
            raise ValueError(f"Process ID '{pid}' already exists")

        process = {
            "pid": pid,
            "arrival": int(payload.get("arrival", 0)),
            "burst": burst,
            "priority": int(payload.get("priority", 0)),
            "io_requests": payload.get("io_requests", []),
        }
        self._processes.append(process)
        return copy.deepcopy(process)

    def clear_processes(self) -> None:
        self._processes.clear()

    def set_last_simulation(self, data: Dict[str, Any]) -> None:
        self._last_simulation = copy.deepcopy(data)

    def get_last_simulation(self) -> Optional[Dict[str, Any]]:
        if self._last_simulation is None:
            return None
        return copy.deepcopy(self._last_simulation)


class SystemMonitor:
    def __init__(self, store: ProcessManager):
        self.store = store
        self.boot_time = time.time()

    def _rng(self, seed: Any) -> random.Random:
        return random.Random(str(seed))

    def snapshot(self) -> Dict[str, Any]:
        processes = self.store.list_processes()
        last_sim = self.store.get_last_simulation() or {}
        now = time.time()

        system_clock = last_sim.get("total") or int(now - self.boot_time)
        total_burst = sum(proc["burst"] for proc in processes) or 1
        cpu_load = min(100, total_burst * 5)
        memory_usage = min(100, (len(processes) * 6) + (cpu_load / 2))

        heap_allocated = int(512 * (memory_usage / 100))
        heap_free = max(0, 512 - heap_allocated)

        stack_usage = []
        for idx, proc in enumerate(processes, start=1):
            rng = self._rng(f"{proc['pid']}-{idx}")
            limit = rng.randint(128, 256)
            used = rng.randint(32, limit)
            stack_usage.append({
                "pid": proc["pid"],
                "used": used,
                "limit": limit
            })

        context_switches = last_sim.get("context_switches", 0)
        total_time = last_sim.get("total", 0) or 1
        context_switch_rate = round(context_switches / total_time, 2)

        system_metrics = last_sim.get("performance", {}).get("system_metrics", {})

        return {
            "system_clock": system_clock,
            "total_tasks": len(processes),
            "cpu": {
                "load_percent": round(cpu_load, 2),
                "utilization": system_metrics.get("cpu_utilization", 0)
            },
            "memory": {
                "usage_percent": round(memory_usage, 2),
                "heap": {
                    "allocated": heap_allocated,
                    "free": heap_free
                },
                "stacks": stack_usage
            },
            "context_switch_rate": context_switch_rate,
            "interrupt_frequency": round(context_switch_rate * 4, 2)
        }

    def ipc_snapshot(self) -> Dict[str, Any]:
        processes = self.store.list_processes()
        names = [proc["pid"] for proc in processes]
        rng = self._rng(len(processes) or 1)

        def waiting_sample():
            if not names:
                return []
            sample_size = rng.randint(0, min(3, len(names)))
            return rng.sample(names, k=sample_size)

        semaphores = [
            {"name": "SEM_TX", "count": rng.randint(0, 3), "waiting": waiting_sample()},
            {"name": "SEM_RX", "count": rng.randint(0, 2), "waiting": waiting_sample()},
        ]

        mutexes = [
            {"name": "MUTEX_SPI", "owner": rng.choice(names) if names else None, "waiting": waiting_sample()},
            {"name": "MUTEX_LOG", "owner": rng.choice(names) if names else None, "waiting": waiting_sample()},
        ]

        queues = [
            {"name": "QUEUE_TELEMETRY", "fill_level": rng.randint(0, 100)},
            {"name": "QUEUE_EVENTS", "fill_level": rng.randint(0, 100)},
        ]

        return {
            "semaphores": semaphores,
            "mutexes": mutexes,
            "queues": queues
        }


process_store = ProcessManager()
system_monitor = SystemMonitor(process_store)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/processes", methods=["GET", "POST", "DELETE"])
def processes_api():
    if request.method == "GET":
        return jsonify({"processes": process_store.list_processes()})

    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        try:
            process = process_store.add_process(payload)
        except (ValueError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"process": process}), 201

    process_store.clear_processes()
    return jsonify({"status": "cleared"}), 200


@app.route("/api/system", methods=["GET"])
def system_snapshot():
    return jsonify(system_monitor.snapshot())


@app.route("/api/ipc", methods=["GET"])
def ipc_snapshot():
    return jsonify(system_monitor.ipc_snapshot())


@app.route("/api/performance", methods=["GET"])
def performance_snapshot():
    data = process_store.get_last_simulation()
    if not data:
        return jsonify({"error": "No simulation data available"}), 404
    return jsonify(data)

@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.get_json() or {}
    processes_data = data.get("processes") or process_store.list_processes()
    if not processes_data:
        return jsonify({"error": "No processes supplied"}), 400

    algo = data.get("algorithm", "FCFS")
    quantum = int(data.get("quantum", 2)) if data.get("quantum") else 2

    processes = []
    for p_data in processes_data:
        io_requests = p_data.get("io_requests", [])
        process = Process(
            pid=p_data["pid"],
            arrival=p_data.get("arrival", 0),
            burst=p_data["burst"],
            priority=p_data.get("priority", 0),
            io_requests=io_requests
        )
        processes.append(process)

    timeline = []
    state_transitions = []
    current_time = 0

    if algo == "FCFS":
        processes.sort(key=lambda x: (x.arrival, x.pid))
        ready_queue = []
        time = 0
        i = 0
        
        while i < len(processes) or ready_queue:
            while i < len(processes) and processes[i].arrival <= time:
                processes[i].add_state_transition(ProcessState.READY, time)
                ready_queue.append(processes[i])
                i += 1
            
            if not ready_queue:
                if i < len(processes):
                    time = processes[i].arrival
                else:
                    break
                continue
            
            current = ready_queue.pop(0)
            current.add_state_transition(ProcessState.RUNNING, time)
            
            run_time = current.remaining_burst
            timeline.append({"pid": current.pid, "start": time, "end": time + run_time})
            current.remaining_burst = 0
            time += run_time
            
            io_occurred = False
            for io_time, io_duration in current.io_requests:
                if io_time < time and io_time + io_duration > time - run_time:
                    current.add_state_transition(ProcessState.WAITING, time)
                    time += io_duration
                    current.add_state_transition(ProcessState.READY, time)
                    ready_queue.append(current)
                    io_occurred = True
                    break
            
            if not io_occurred:
                current.add_state_transition(ProcessState.TERMINATED, time)

    elif algo == "SJF":
        processes.sort(key=lambda x: x.arrival)
        ready_queue = []
        time = 0
        i = 0
        
        while i < len(processes) or ready_queue:
            while i < len(processes) and processes[i].arrival <= time:
                processes[i].add_state_transition(ProcessState.READY, time)
                ready_queue.append(processes[i])
                i += 1
            
            if not ready_queue:
                if i < len(processes):
                    time = processes[i].arrival
                else:
                    break
                continue
            
            ready_queue.sort(key=lambda x: x.remaining_burst)
            current = ready_queue.pop(0)
            current.add_state_transition(ProcessState.RUNNING, time)
            
            run_time = current.remaining_burst
            timeline.append({"pid": current.pid, "start": time, "end": time + run_time})
            current.remaining_burst = 0
            time += run_time
            
            io_occurred = False
            for io_time, io_duration in current.io_requests:
                if io_time < time and io_time + io_duration > time - run_time:
                    current.add_state_transition(ProcessState.WAITING, time)
                    time += io_duration
                    current.add_state_transition(ProcessState.READY, time)
                    ready_queue.append(current)
                    io_occurred = True
                    break
            
            if not io_occurred:
                current.add_state_transition(ProcessState.TERMINATED, time)

    elif algo == "SRTF":
        processes.sort(key=lambda x: x.arrival)
        ready_queue = []
        time = 0
        i = 0
        current_running = None
        context_switches = 0
        
        while i < len(processes) or ready_queue or current_running:
            while i < len(processes) and processes[i].arrival <= time:
                processes[i].add_state_transition(ProcessState.READY, time)
                ready_queue.append(processes[i])
                i += 1
            
            ready_queue.sort(key=lambda x: x.remaining_burst)
            
            if current_running and ready_queue:
                if ready_queue[0].remaining_burst < current_running.remaining_burst:
                    current_running.add_state_transition(ProcessState.READY, time)
                    ready_queue.append(current_running)
                    context_switches += 1
                    current_running = None
            
            if not current_running and ready_queue:
                current_running = ready_queue.pop(0)
                if current_running.state != ProcessState.RUNNING:
                    current_running.add_state_transition(ProcessState.RUNNING, time)
            
            if not current_running:
                if i < len(processes):
                    time = processes[i].arrival
                else:
                    break
                continue
            
            timeline.append({"pid": current_running.pid, "start": time, "end": time + 1})
            current_running.remaining_burst -= 1
            time += 1
            
            io_occurred = False
            for io_time, io_duration in current_running.io_requests:
                if io_time <= time and io_time + io_duration > time:
                    current_running.add_state_transition(ProcessState.WAITING, time)
                    time += io_duration
                    current_running.add_state_transition(ProcessState.READY, time)
                    ready_queue.append(current_running)
                    current_running = None
                    io_occurred = True
                    break
            
            if not io_occurred and current_running.remaining_burst == 0:
                current_running.add_state_transition(ProcessState.TERMINATED, time)
                current_running = None

    elif algo == "Priority":
        processes.sort(key=lambda x: x.arrival)
        ready_queue = []
        time = 0
        i = 0
        
        while i < len(processes) or ready_queue:
            while i < len(processes) and processes[i].arrival <= time:
                processes[i].add_state_transition(ProcessState.READY, time)
                ready_queue.append(processes[i])
                i += 1
            
            if not ready_queue:
                if i < len(processes):
                    time = processes[i].arrival
                else:
                    break
                continue
            
            ready_queue.sort(key=lambda x: x.priority)
            current = ready_queue.pop(0)
            current.add_state_transition(ProcessState.RUNNING, time)
            
            run_time = current.remaining_burst
            timeline.append({"pid": current.pid, "start": time, "end": time + run_time})
            current.remaining_burst = 0
            time += run_time
            
            io_occurred = False
            for io_time, io_duration in current.io_requests:
                if io_time < time and io_time + io_duration > time - run_time:
                    current.add_state_transition(ProcessState.WAITING, time)
                    time += io_duration
                    current.add_state_transition(ProcessState.READY, time)
                    ready_queue.append(current)
                    io_occurred = True
                    break
            
            if not io_occurred:
                current.add_state_transition(ProcessState.TERMINATED, time)

    elif algo == "Round Robin":
        processes.sort(key=lambda x: x.arrival)
        ready_queue = deque()
        time = 0
        i = 0
        context_switches = 0
        
        while i < len(processes) or ready_queue:
            while i < len(processes) and processes[i].arrival <= time:
                processes[i].add_state_transition(ProcessState.READY, time)
                ready_queue.append(processes[i])
                i += 1
            
            if not ready_queue:
                if i < len(processes):
                    time = processes[i].arrival
                else:
                    break
                continue
            
            current = ready_queue.popleft()
            if current.state != ProcessState.RUNNING:
                current.add_state_transition(ProcessState.RUNNING, time)
            
            run_time = min(quantum, current.remaining_burst)
            timeline.append({"pid": current.pid, "start": time, "end": time + run_time})
            current.remaining_burst -= run_time
            time += run_time
            
            while i < len(processes) and processes[i].arrival <= time:
                processes[i].add_state_transition(ProcessState.READY, time)
                ready_queue.append(processes[i])
                i += 1
            
            io_occurred = False
            for io_time, io_duration in current.io_requests:
                if io_time <= time and io_time + io_duration > time:
                    current.add_state_transition(ProcessState.WAITING, time)
                    time += io_duration
                    current.add_state_transition(ProcessState.READY, time)
                    ready_queue.append(current)
                    io_occurred = True
                    break
            
            if not io_occurred:
                if current.remaining_burst > 0:
                    current.add_state_transition(ProcessState.READY, time)
                    ready_queue.append(current)
                    context_switches += 1
                else:
                    current.add_state_transition(ProcessState.TERMINATED, time)

    total_time = time
    
    all_transitions = []
    for process in processes:
        all_transitions.extend(process.state_history)
    
    all_transitions.sort(key=lambda x: x['time'])
    
    def calculate_metrics(processes, timeline):
        metrics = []
        for p in processes:
            finish_times = [seg["end"] for seg in timeline if seg["pid"] == p.pid]
            finish_time = max(finish_times) if finish_times else 0
            
            first_cpu_times = [seg["start"] for seg in timeline if seg["pid"] == p.pid]
            first_cpu = min(first_cpu_times) if first_cpu_times else 0
            
            arrival = p.arrival
            burst = p.burst
            
            turnaround_time = finish_time - arrival
            waiting_time = turnaround_time - burst
            response_time = first_cpu - arrival
            
            metrics.append({
                "pid": p.pid,
                "arrival": arrival,
                "burst": burst,
                "finish": finish_time,
                "turnaround": turnaround_time,
                "waiting": waiting_time,
                "response": response_time
            })
        
        avg_waiting = sum(m["waiting"] for m in metrics) / len(metrics) if metrics else 0
        avg_turnaround = sum(m["turnaround"] for m in metrics) / len(metrics) if metrics else 0
        avg_response = sum(m["response"] for m in metrics) / len(metrics) if metrics else 0
        
        cpu_time = sum(seg["end"] - seg["start"] for seg in timeline if seg["pid"] != "IDLE")
        cpu_utilization = (cpu_time / total_time * 100) if total_time > 0 else 0
        
        throughput = len(processes) / total_time if total_time > 0 else 0
        
        return {
            "process_metrics": metrics,
            "averages": {
                "waiting": round(avg_waiting, 2),
                "turnaround": round(avg_turnaround, 2),
                "response": round(avg_response, 2)
            },
            "system_metrics": {
                "cpu_utilization": round(cpu_utilization, 2),
                "throughput": round(throughput, 3)
            }
        }
    
    performance_metrics = calculate_metrics(processes, timeline)
    
    response_payload = {
        "timeline": timeline, 
        "total": total_time,
        "performance": performance_metrics,
        "state_transitions": all_transitions,
        "context_switches": context_switches if 'context_switches' in locals() else 0
    }
    process_store.set_last_simulation(response_payload)
    
    return jsonify(response_payload)


if __name__ == "__main__":
    app.run(debug=True)
