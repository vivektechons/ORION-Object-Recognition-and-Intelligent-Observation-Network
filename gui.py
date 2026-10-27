"""
gui.py - Tkinter control panel (dark dashboard style). Presentation only: it never touches
hardware directly and has no business logic.

All blocking work (opening the camera, which can take seconds) runs in short worker
threads; Tk widgets are only touched from the main thread. The panel refreshes from an
immutable AppSnapshot every ~33 ms.

Run:   started by main.py (python main.py)
Deps:  tkinter (bundled with python.org installers). No import of main.py (no cycles).
Keyboard: Esc = EMERGENCY STOP.
Look:  uses only the standard library. Plain Tkinter cannot draw rounded corners, so cards
       are flat with a thin coloured border. Icons are Unicode symbols (Segoe UI Emoji/Symbol).
"""
import threading
import tkinter as tk
from tkinter import ttk

# ------------------------------------------------------------------ palette
BG = "#070d18"
CARD = "#0d1727"
FIELD = "#0a1424"
BORDER = "#1b3556"
TEXT = "#e6edf7"
MUTED = "#8ea3bf"
ACCENT = "#14c8ff"
GREEN = "#22d46b"
AMBER = "#ffb02e"
RED = "#ff3b4e"
BLUE = "#4aa3ff"

FONT = "Segoe UI"
ICON_FONT = "Segoe UI Emoji"

# banner: (background, border, accent colour)
LEVEL_STYLE = {
    "critical": ("#3a0710", RED, "#ff8a96"),
    "warn": ("#3a2200", "#c77a12", AMBER),
    "info": ("#08243f", BLUE, "#8cc4ff"),
    "ok": ("#06301c", GREEN, "#8ff0b4"),
}
LOG_COLORS = {"DEBUG": GREEN, "INFO": BLUE, "WARNING": AMBER, "ERROR": RED, "CRITICAL": RED}

STATE_COLORS = {
    "NO LINK": RED, "NO TELEMETRY": RED, "ESTOP": RED, "FAULT": RED,
    "DISARMED": AMBER, "TRACKING": ACCENT, "READY": RED, "ACTIVATING": RED, "LOCKOUT": BLUE,
}
DETECT_COLORS = {"TARGET": GREEN, "NO_TARGET": MUTED, "REJECTED": AMBER}


def _hint(s) -> str:
    """Right-hand hint line of the banner."""
    if s.serial_state != "CONNECTED":
        return "Connect ESP32 to enable tracking and robotic arm control."
    return {
        "ESTOP": "Release the physical e-stop, then press RESET.",
        "FAULT": "Fix the cause of the fault, then press RESET.",
        "DISARMED": "Press ARM when the area is clear to enable the actuator.",
        "TRACKING": "Armed: the actuator fires once after the stability time.",
        "READY": "Activation is imminent.",
        "ACTIVATING": "Actuator moving.",
        "LOCKOUT": "Fired once. Move the object away (or press Re-arm) to enable the next pop.",
    }.get(s.esp_state, "")


class ControlPanel:
    REFRESH_MS = 33

    def __init__(self, root: tk.Tk, app, cfg):
        self.root = root
        self.app = app
        self.cfg = cfg
        self._busy_camera = False
        self._last_frame_id = -1
        self._photo = None
        self._log_cache = None
        self._log_marker = None            # last message hidden by the Clear button
        self._banner_key = None
        self._closing = False
        self._btn_colors = {}

        root.title("ORION — Object Recognition and Intelligent Observation Network")
        root.configure(bg=BG)
        root.minsize(1100, 700)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind_all("<Escape>", lambda _e: self.app.emergency_stop())
        self._style()
        self._build()
        self._refresh_ports()
        self._tick()

    # ---------------------------------------------------------------- styling
    def _style(self) -> None:
        r = self.root
        st = ttk.Style(r)
        st.theme_use("clam")
        st.configure("Dark.TCombobox", fieldbackground=FIELD, background=FIELD, foreground=TEXT,
                     arrowcolor=ACCENT, bordercolor=BORDER, lightcolor=FIELD, darkcolor=FIELD,
                     selectbackground=FIELD, selectforeground=TEXT, padding=4)
        st.map("Dark.TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", TEXT)],
               selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", TEXT)])
        st.configure("Dark.TSpinbox", fieldbackground=FIELD, background=FIELD, foreground=TEXT,
                     arrowcolor=ACCENT, bordercolor=BORDER, lightcolor=FIELD, darkcolor=FIELD, padding=3)
        st.configure("Cyan.Horizontal.TProgressbar", troughcolor=FIELD, background=ACCENT,
                     bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT, thickness=8)
        st.configure("Dark.Vertical.TScrollbar", troughcolor=FIELD, background=BORDER, bordercolor=FIELD,
                     arrowcolor=MUTED, lightcolor=BORDER, darkcolor=BORDER)
        r.option_add("*TCombobox*Listbox.background", FIELD)
        r.option_add("*TCombobox*Listbox.foreground", TEXT)
        r.option_add("*TCombobox*Listbox.selectBackground", BORDER)
        r.option_add("*TCombobox*Listbox.selectForeground", TEXT)

    def _card(self, parent, title: str, icon: str, border: str = BORDER):
        """Returns (outer frame, body frame, header frame)."""
        outer = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=border, highlightcolor=border)
        head = tk.Frame(outer, bg=CARD)
        head.pack(fill="x", padx=10, pady=(5, 2))
        tk.Label(head, text=icon, bg=CARD, fg=ACCENT, font=(ICON_FONT, 13)).pack(side="left")
        tk.Label(head, text=title, bg=CARD, fg=TEXT, font=(FONT, 11, "bold")).pack(side="left", padx=(6, 0))
        body = tk.Frame(outer, bg=CARD)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 7))
        return outer, body, head

    def _button(self, parent, text, command, bg, fg=TEXT, border=BORDER, hover=None, font=None, **kw):
        b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg, activebackground=hover or bg,
                      activeforeground=fg, relief="flat", bd=0, cursor="hand2", highlightthickness=1,
                      highlightbackground=border, highlightcolor=border, font=font or (FONT, 10, "bold"),
                      padx=8, pady=4, **kw)
        self._btn_colors[b] = (bg, hover or bg)
        b.bind("<Enter>", lambda _e, w=b: w.config(bg=self._btn_colors[w][1]))
        b.bind("<Leave>", lambda _e, w=b: w.config(bg=self._btn_colors[w][0]))
        return b

    def _restyle(self, b, bg, fg, border, hover) -> None:
        self._btn_colors[b] = (bg, hover)
        b.config(bg=bg, fg=fg, activebackground=hover, activeforeground=fg,
                 highlightbackground=border, highlightcolor=border)

    # ----------------------------------------------------------------- layout
    def _build(self) -> None:
        r = self.root
        r.columnconfigure(0, weight=1)
        r.rowconfigure(1, weight=1)

        # ---- banner
        self.banner = tk.Frame(r, bg="#3a2200", highlightthickness=1, highlightbackground="#c77a12")
        self.banner.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 6))
        self.b_icon = tk.Label(self.banner, text="\u26a0", font=(ICON_FONT, 20), bg="#3a2200", fg=AMBER)
        self.b_icon.pack(side="left", padx=(12, 8), pady=6)
        self.b_title = tk.Label(self.banner, text="", font=(FONT, 14, "bold"), bg="#3a2200", fg=TEXT)
        self.b_title.pack(side="left")
        self.b_sub = tk.Label(self.banner, text="", font=(FONT, 14, "bold"), bg="#3a2200", fg=AMBER)
        self.b_sub.pack(side="left", padx=(6, 0))
        self._button(self.banner, "\u2715  Stop and exit", self.on_close, FIELD, MUTED, BORDER, BORDER,
                     font=(FONT, 9)).pack(side="right", padx=(0, 12))
        self.b_hint = tk.Label(self.banner, text="", font=(FONT, 10), bg="#3a2200", fg=TEXT)
        self.b_hint.pack(side="right", padx=14)

        # ---- left column
        left = tk.Frame(r, bg=BG)
        left.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=(0, 10))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        vcard, vbody, vhead = self._card(left, "Live Camera Feed", "\U0001F4F7", ACCENT)
        vcard.grid(row=0, column=0, sticky="ew")
        self.fps_dot = tk.Label(vhead, text="\u25cf", bg=CARD, fg=RED, font=(FONT, 11))
        self.fps_lbl = tk.Label(vhead, text="FPS: 0.0", bg=CARD, fg=TEXT, font=(FONT, 10))
        self.fps_lbl.pack(side="right")
        self.fps_dot.pack(side="right", padx=(0, 4))
        self.cam_info = tk.Label(vhead, text="Camera: not connected", bg=CARD, fg=MUTED, font=(FONT, 9))
        self.cam_info.pack(side="right", padx=(0, 16))
        self.video = tk.Label(vbody, text="No camera connected", width=80, height=24, bg="#05090f", fg=MUTED,
                              font=(FONT, 11))
        self.video.pack()

        lcard, lbody, lhead = self._card(left, "Debug Log", "\u276f_")
        lcard.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.autoscroll = tk.BooleanVar(value=True)
        tk.Checkbutton(lhead, text="Auto Scroll", variable=self.autoscroll, bg=CARD, fg=TEXT, selectcolor=FIELD,
                       activebackground=CARD, activeforeground=TEXT, font=(FONT, 9), bd=0,
                       highlightthickness=0).pack(side="right")
        self._button(lhead, "\U0001F5D1  Clear", self._clear_log, FIELD, TEXT, BORDER, BORDER,
                     font=(FONT, 9)).pack(side="right", padx=8)
        logrow = tk.Frame(lbody, bg=CARD)
        logrow.pack(fill="both", expand=True)
        self.log_text = tk.Text(logrow, height=6, state="disabled", font=("Consolas", 9), wrap="none",
                                bg="#08111f", fg=TEXT, bd=0, highlightthickness=0, insertbackground=TEXT)
        sb = ttk.Scrollbar(logrow, orient="vertical", command=self.log_text.yview, style="Dark.Vertical.TScrollbar")
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_text.tag_configure("time", foreground=MUTED)
        self.log_text.tag_configure("msg", foreground=TEXT)
        for lvl, col in LOG_COLORS.items():
            self.log_text.tag_configure(lvl, foreground=col)

        # ---- right column
        right = tk.Frame(r, bg=BG)
        right.grid(row=1, column=1, sticky="ns", padx=(5, 10), pady=(0, 10))

        ccard, cbody, _ = self._card(right, "Camera Settings", "\U0001F4F7", ACCENT)
        ccard.pack(fill="x")
        cbody.columnconfigure(1, weight=1)
        cbody.columnconfigure(3, weight=1)
        tk.Label(cbody, text="Index", bg=CARD, fg=TEXT, font=(FONT, 10)).grid(row=0, column=0, sticky="w")
        self.cam_index = tk.StringVar(value=str(self.cfg.camera.index))
        ttk.Combobox(cbody, values=[str(i) for i in range(10)], width=5, state="readonly",
                     textvariable=self.cam_index, style="Dark.TCombobox").grid(row=0, column=1, sticky="ew", padx=(6, 12))
        tk.Label(cbody, text="Backend", bg=CARD, fg=TEXT, font=(FONT, 10)).grid(row=0, column=2, sticky="w")
        self.cam_backend = tk.StringVar(value=self.cfg.camera.backend)
        ttk.Combobox(cbody, values=self.app.backend_names(), width=8, state="readonly",
                     textvariable=self.cam_backend, style="Dark.TCombobox").grid(row=0, column=3, sticky="ew", padx=(6, 0))
        self.cam_btn = self._button(cbody, "\U0001F4F7  Connect Camera", self._toggle_camera, ACCENT, "#04202b", ACCENT, "#59dcff")
        self.cam_btn.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        scard, sbody, shead = self._card(right, "ESP32 Connection", "\u2699")
        self.ser_badge = tk.Label(shead, text="\u25cb not connected", bg=CARD, fg=MUTED, font=(FONT, 9, "bold"))
        self.ser_badge.pack(side="right")
        scard.pack(fill="x", pady=(6, 0))
        sbody.columnconfigure(0, weight=1)
        self.port_var = tk.StringVar(value=self.cfg.serial.port)
        self.port_box = ttk.Combobox(sbody, width=14, textvariable=self.port_var, state="readonly",
                                     style="Dark.TCombobox")
        self.port_box.grid(row=0, column=0, sticky="ew")
        self._button(sbody, "\u21bb", self._refresh_ports, FIELD, TEXT, BORDER, BORDER,
                     font=(FONT, 12, "bold")).grid(row=0, column=1, padx=6)
        self.ser_btn = self._button(sbody, "\U0001F517  Connect ESP32", self._toggle_serial, ACCENT, "#04202b", ACCENT, "#59dcff")
        self.ser_btn.grid(row=0, column=2)

        tcard, tbody, _ = self._card(right, "Tracking & Control", "\u2699")
        tcard.pack(fill="x", pady=(6, 0))
        tbody.columnconfigure(0, weight=1, uniform="c")
        tbody.columnconfigure(1, weight=1, uniform="c")
        self.track_btn = self._button(tbody, "\u25b6  Tracking: OFF (click to enable)", self._toggle_tracking,
                                      FIELD, TEXT, BORDER, BORDER)
        self.track_btn.grid(row=0, column=0, columnspan=2, sticky="ew")
        row1 = tk.Frame(tbody, bg=CARD)
        row1.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        tk.Label(row1, text="Stability time (s)", bg=CARD, fg=TEXT, font=(FONT, 10)).pack(side="left")
        self.stab_var = tk.StringVar(value=f"{self.cfg.stability.required_s:g}")
        ttk.Spinbox(row1, from_=0.5, to=60, increment=0.5, width=6, textvariable=self.stab_var,
                    style="Dark.TSpinbox").pack(side="left", padx=8)
        self._button(row1, "Set", self._set_stability, FIELD, TEXT, BORDER, BORDER, font=(FONT, 9, "bold")).pack(side="left")
        self._button(tbody, "\U0001F6E1  ARM", self.app.arm, "#0b3b3d", "#7ff3e6", "#16a39a", "#10575a"
                     ).grid(row=2, column=0, sticky="ew", pady=(8, 0), padx=(0, 4))
        self._button(tbody, "\U0001F6E1  DISARM", self.app.disarm, "#3a0f1b", "#ff9aa8", "#a02a40", "#561728"
                     ).grid(row=2, column=1, sticky="ew", pady=(8, 0), padx=(4, 0))
        row3 = tk.Frame(tbody, bg=CARD)
        row3.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        for col in range(3):
            row3.columnconfigure(col, weight=1, uniform="r3")
        self._button(row3, "\u2302  Home", self.app.go_home, FIELD, TEXT, BORDER, BORDER
                     ).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self._button(row3, "\u21ba  RESET", self.app.reset, FIELD, TEXT, BORDER, BORDER
                     ).grid(row=0, column=1, sticky="ew", padx=3)
        self._button(row3, "\u27f3  Re-arm", self.app.rearm, "#10304a", "#8cc4ff", BLUE, "#16486f"
                     ).grid(row=0, column=2, sticky="ew", padx=(3, 0))
        self._button(tbody, "\u25a0  EMERGENCY STOP  (Esc)", self.app.emergency_stop, "#d50000", "white", "#ff5a5a",
                     "#ff1f1f", font=(FONT, 12, "bold")).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0), ipady=2)

        pcard, pbody, _ = self._card(right, "Status & Telemetry", "\U0001F4CA")
        pcard.pack(fill="x", pady=(6, 0))
        pbody.columnconfigure(1, weight=1)
        self.vars = {}
        self.val_labels = {}
        keys = ["State", "Detection", "Target", "Error", "Stability", "Pan", "Tilt", "Actuator", "FPS"]
        for i, key in enumerate(keys):
            tk.Label(pbody, text=key + ":", bg=CARD, fg=TEXT, font=(FONT, 10)).grid(row=i, column=0, sticky="w")
            v = tk.StringVar(value="-")
            lbl = tk.Label(pbody, textvariable=v, bg=CARD, fg=TEXT, font=(FONT, 10), anchor="w", width=34)
            lbl.grid(row=i, column=1, sticky="w", padx=(10, 0))
            self.vars[key] = v
            self.val_labels[key] = lbl
        self.progress = ttk.Progressbar(pbody, maximum=1.0, style="Cyan.Horizontal.TProgressbar")
        self.progress.grid(row=len(keys), column=0, columnspan=2, sticky="ew", pady=(6, 4))

    # ---------------------------------------------------------------- actions
    def _bg(self, fn) -> None:
        threading.Thread(target=fn, daemon=True).start()

    def _toggle_camera(self) -> None:
        if self._busy_camera:
            return
        snap = self.app.get_snapshot()
        if snap.camera_ok:
            self._bg(self.app.disconnect_camera)
            return
        try:
            index = int(self.cam_index.get())
        except ValueError:
            self.app.log.add("ERROR", "Camera index must be a whole number")
            return
        backend = self.cam_backend.get()
        self._busy_camera = True
        self.cam_btn.config(text="Connecting...", state="disabled")

        def work():
            try:
                self.app.connect_camera(index, backend)
            finally:
                self._busy_camera = False
        self._bg(work)

    def _refresh_ports(self) -> None:
        ports = self.app.serial_ports()
        names = [p for p, _d in ports]
        self.port_box["values"] = names
        if self.port_var.get() not in names:
            self.port_var.set(names[0] if names else "")

    def _toggle_serial(self) -> None:
        snap = self.app.get_snapshot()
        if snap.serial_state in ("CONNECTED", "CONNECTING"):
            self._bg(self.app.disconnect_serial)
        else:
            port = self.port_var.get()
            if not port:
                self.app.log.add("ERROR", "No COM port selected (plug in the ESP32 and press refresh)")
                return
            self.app.connect_serial(port)

    def _toggle_tracking(self) -> None:
        self.app.set_tracking(not self.app.get_snapshot().tracking_enabled)

    def _set_stability(self) -> None:
        try:
            ok, msg = self.app.set_required_stability(float(self.stab_var.get()))
        except ValueError:
            ok, msg = False, "Stability time must be a number"
        if not ok:
            self.app.log.add("WARNING", msg)

    def _clear_log(self) -> None:
        msgs = self.app.recent_messages(60)
        self._log_marker = msgs[-1] if msgs else None
        self._log_cache = None

    # ---------------------------------------------------------------- refresh
    def _update_banner(self, s) -> None:
        key = (s.warning, s.level, _hint(s))
        if key == self._banner_key:
            return
        self._banner_key = key
        bg, border, accent = LEVEL_STYLE.get(s.level, LEVEL_STYLE["warn"])
        head, sep, tail = s.warning.partition(": ")
        title = head + (":" if sep else "")
        for w in (self.banner, self.b_icon, self.b_title, self.b_sub, self.b_hint):
            w.config(bg=bg)
        self.banner.config(highlightbackground=border, highlightcolor=border)
        self.b_icon.config(fg=accent)
        self.b_title.config(text=title, fg=TEXT)
        self.b_sub.config(text=tail, fg=accent)
        self.b_hint.config(text=("\u24d8  " + _hint(s)) if _hint(s) else "", fg=TEXT)

    def _update_video(self, s) -> None:
        if s.ppm is not None and s.frame_id != self._last_frame_id:
            self._last_frame_id = s.frame_id
            try:
                self._photo = tk.PhotoImage(data=s.ppm, format="PPM")
                self.video.config(image=self._photo, text="", width=0, height=0)
            except tk.TclError:
                pass
        elif s.ppm is None and self._last_frame_id != -2:
            self._last_frame_id = -2
            self._photo = None
            self.video.config(image="", text="No camera frames", width=80, height=24)
        self.fps_lbl.config(text=f"FPS: {s.fps:.1f}")
        self.fps_dot.config(fg=GREEN if s.fps >= 15 else (AMBER if s.fps > 0 else RED))

    def _update_log(self) -> None:
        msgs = self.app.recent_messages(60)
        if self._log_marker is not None:
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i] == self._log_marker:
                    msgs = msgs[i + 1:]
                    break
        if msgs == self._log_cache:
            return
        self._log_cache = msgs
        top = self.log_text.yview()[0]
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        for ts, lvl, msg in msgs:
            self.log_text.insert("end", "\u25cf ", lvl)
            self.log_text.insert("end", f"{ts}  ", "time")
            self.log_text.insert("end", f"{lvl:<8} ", lvl)
            self.log_text.insert("end", msg + "\n", "msg")
        if self.autoscroll.get():
            self.log_text.see("end")
        else:
            self.log_text.yview_moveto(top)
        self.log_text.config(state="disabled")

    def _tick(self) -> None:
        if self._closing:
            return
        s = self.app.get_snapshot()
        self._update_banner(s)
        self._update_video(s)

        v, lab = self.vars, self.val_labels
        v["State"].set(f"{s.app_state}   (armed={'YES' if s.armed else 'no'})")
        lab["State"].config(fg=STATE_COLORS.get(s.app_state, TEXT))
        v["Detection"].set(s.detection_status)
        lab["Detection"].config(fg=DETECT_COLORS.get(s.detection_status, TEXT))
        v["Target"].set(s.target_text)
        v["Error"].set(s.error_text)
        v["Stability"].set(f"{s.stable_s:.1f} / {s.required_s:.1f} s  [{s.stability_status}]")
        rep = lambda x: "-" if x is None else f"{x:.1f}"
        v["Pan"].set(f"cmd {s.pan_cmd:.1f}   reported {rep(s.pan_rep)} deg")
        v["Tilt"].set(f"cmd {s.tilt_cmd:.1f}   reported {rep(s.tilt_rep)} deg")
        v["Actuator"].set(s.actuator_text)
        lab["Actuator"].config(fg=RED if s.actuator_text.startswith("ACTIVATING") else TEXT)
        v["FPS"].set(f"{s.fps:.1f} (measured)   detect {s.detect_ms:.0f} ms   loop {s.proc_ms:.0f} ms")
        self.progress["value"] = s.progress
        cam_txt = s.camera_text if len(s.camera_text) <= 60 else s.camera_text[:57] + "..."
        self.cam_info.config(text=cam_txt, fg=RED if "ERROR" in s.camera_text else MUTED)
        badge = {"CONNECTED": ("\u25cf connected", GREEN), "CONNECTING": ("\u25cf connecting...", AMBER),
                 "ERROR": ("\u25cf error (see log)", RED)}.get(s.serial_state, ("\u25cb not connected", MUTED))
        self.ser_badge.config(text=badge[0], fg=badge[1])

        if not self._busy_camera:
            if s.camera_ok:
                self._restyle(self.cam_btn, FIELD, TEXT, BORDER, BORDER)
                self.cam_btn.config(text="\U0001F4F7  Disconnect Camera", state="normal")
            else:
                self._restyle(self.cam_btn, ACCENT, "#04202b", ACCENT, "#59dcff")
                self.cam_btn.config(text="\U0001F4F7  Connect Camera", state="normal")
        if s.serial_state == "CONNECTED":
            self._restyle(self.ser_btn, FIELD, TEXT, BORDER, BORDER)
            self.ser_btn.config(text="\U0001F517  Disconnect ESP32")
        elif s.serial_state == "CONNECTING":
            self._restyle(self.ser_btn, FIELD, AMBER, AMBER, BORDER)
            self.ser_btn.config(text="Connecting... (cancel)")
        else:
            self._restyle(self.ser_btn, ACCENT, "#04202b", ACCENT, "#59dcff")
            self.ser_btn.config(text="\U0001F517  Connect ESP32")
        if s.tracking_enabled:
            self._restyle(self.track_btn, "#0e3d2b", "#bff5d6", GREEN, "#14573d")
            self.track_btn.config(text="\u25b6  Tracking: ON (click to disable)")
        else:
            self._restyle(self.track_btn, FIELD, TEXT, BORDER, BORDER)
            self.track_btn.config(text="\u25b6  Tracking: OFF (click to enable)")

        self._update_log()
        self.root.after(self.REFRESH_MS, self._tick)

    def on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        try:
            self.app.shutdown()
        finally:
            self.root.destroy()
