"""
Esmini simulator process management.

Handles starting and stopping the esmini simulator process.
"""

import subprocess
import time
import os


class EsminiRunner:
    """Manages the esmini simulator process lifecycle."""
    
    def __init__(self, exe_path: str, scenario_path: str):
        """
        Initialize the esmini runner.
        
        Args:
            exe_path: Path to the esmini executable (relative or absolute)
            scenario_path: Path to the OpenSCENARIO file to run (relative or absolute)
        """
        # Resolve paths - if relative, make them absolute from project root
        # Get project root: src/sim/esmini_runner.py -> go up 3 levels to project root
        current_file = os.path.abspath(__file__)
        # __file__ is at src/sim/esmini_runner.py, so:
        # dirname 1: src/sim/
        # dirname 2: src/
        # dirname 3: project root (r_p/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        
        if not os.path.isabs(exe_path):
            # Normalize path separators and join
            normalized_path = exe_path.replace('/', os.sep).replace('\\', os.sep)
            self.exe_path = os.path.normpath(os.path.join(project_root, normalized_path))
        else:
            self.exe_path = os.path.normpath(exe_path)
            
        scenario_path = str(scenario_path).strip() if scenario_path else ""
        if not scenario_path:
            raise ValueError("scenario_path is required for esmini (--osc). Check config sim.scenario_path.")
        if not os.path.isabs(scenario_path):
            normalized_path = scenario_path.replace('/', os.sep).replace('\\', os.sep)
            self.scenario_path = os.path.normpath(os.path.join(project_root, normalized_path))
        else:
            self.scenario_path = os.path.normpath(scenario_path)

        # Verify paths exist
        if not os.path.exists(self.exe_path):
            raise FileNotFoundError(f"esmini executable not found at: {self.exe_path}")
        if not os.path.exists(self.scenario_path):
            raise FileNotFoundError(f"Scenario file not found at: {self.scenario_path}")
            
        self.proc = None

    def start(self, headless: bool = False, window_size: str = None):
        """
        Start the esmini simulator process.
        
        Args:
            headless: If True, run without visualization window
            window_size: Window size as "x y width height" (e.g., "60 60 1280 720")
        
        Note: Adjust command-line arguments based on your esmini version
        and requirements (e.g., UDP ports, window mode, etc.).
        """
        # Run from esmini bin so scenario paths (../xodr, ../models) resolve
        cwd = os.path.dirname(self.exe_path)
        # Use path esmini can parse (forward slashes avoid "missing filename" on some builds)
        osc_arg = self.scenario_path.replace("\\", "/")
        args = [self.exe_path, "--osc", osc_arg]
        if headless:
            args.append("--headless")
        else:
            # So UI can receive our control (throttle/brake/steer) and match logs
            args.append("--player_server")
            if window_size:
                args.extend(["--window", window_size])
        self.proc = subprocess.Popen(args, cwd=cwd)

        # Wait a bit so esmini initializes and UDP starts sending
        time.sleep(2.0)

    def stop(self):
        """
        Stop the esmini simulator process gracefully.
        
        First attempts termination, then kills if necessary.
        """
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
