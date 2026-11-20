"""Keithley 2450 driver for ABAA junction tuning (SCPI command set).

Every burst and I-V runs as an instrument-timed list sweep, so timing does not
depend on the PC or the LAN.

Behaviour measured on the Hatlab 2450 (firmware 1.7.16a, Nov 2026):
  - a list sweep turns the output OFF when it ends; _list_sweep turns it back
    on at 0 V
  - with the output on, sweep setup replays the whole list at about 0.2 ms per
    point (seen on a scope); _list_sweep turns the output off first
  - sweep overhead is about 5.2 ms per point at NPLC 0.05 (T_OVERHEAD below)
  - :SOUR:SWE:VOLT:LIST takes 5 arguments, the last being the buffer name
"""
import numpy as np
import pandas as pd
import pyvisa

I_RANGES = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

T_OVERHEAD = 0.0052   # s per list point, measured from instrument timestamps


def pick_range(i):
    return next(r for r in I_RANGES if r >= abs(i))


class K2450:
    def __init__(self, resource, four_wire=False, terminals="FRON",
                 m_pulses=6, t_pulse=0.200, t_blank=0.100, dt_point=0.025,
                 nplc_burst=0.05, v_iv=0.013, n_iv=27, nplc_iv=1.0, delay_iv=0.005):
        self.four_wire = four_wire
        self.terminals = terminals      # "FRON" or "REAR"
        self.m_pulses = m_pulses
        self.t_pulse = t_pulse
        self.t_blank = t_blank
        self.dt_point = dt_point
        self.nplc_burst = nplc_burst
        self.v_iv = v_iv
        self.n_iv = n_iv
        self.nplc_iv = nplc_iv
        self.delay_iv = delay_iv

        self.rm = pyvisa.ResourceManager()
        self.inst = self.rm.open_resource(resource)
        self.inst.timeout = 30_000
        self.inst.write_termination = "\n"
        self.inst.read_termination = "\n"
        self.ilim = None
        self.timing_reported = False
        print(self.q("*IDN?"))

    def w(self, cmd):
        self.inst.write(cmd)

    def q(self, cmd):
        return self.inst.query(cmd).strip()

    # ---- basic state ----
    def configure(self, ilim=1e-3):
        self.w("*CLS")
        self.w("*RST")
        self.w(f":ROUT:TERM {self.terminals}")
        self.w(":SOUR:FUNC VOLT")
        self.w(":SOUR:VOLT:RANG 2")
        self.w(":SOUR:VOLT:READ:BACK ON")
        self.w(':SENS:FUNC "CURR"')
        self.w(f":SENS:CURR:AZER OFF")
        self.w(f":SENS:CURR:RSEN {'ON' if self.four_wire else 'OFF'}")
        self.w(":OUTP:VOLT:SMOD NORM")       # output-off state: 0 V
        self.set_ilim(ilim)
        self.set_v(0.0)
        self.check_errors()

    def set_ilim(self, ilim):
        self.ilim = float(ilim)
        self.w(f":SOUR:VOLT:ILIM {self.ilim:.6g}")

    def set_v(self, v):
        self.w(f":SOUR:VOLT {v:.6f}")

    def set_range(self, r):
        self.w(f":SOUR:VOLT:RANG {r}")

    def output(self, on):
        self.w(":OUTP ON" if on else ":OUTP OFF")

    def read(self):
        i, v = (float(x) for x in self.q(':READ? "defbuffer1", READ, SOUR').split(","))
        return v, i

    def in_compliance(self, i):
        return abs(i) >= 0.98 * self.ilim

    def check_errors(self):
        errs = []
        for _ in range(50):
            e = self.q(":SYST:ERR?")
            if e.startswith("0,"):
                break
            errs.append(e)
        if errs:
            print("SMU errors:", errs)
        return errs

    def safe(self):
        try:
            self.w(":ABOR")
            self.set_v(0.0)
            self.set_range(2)
        except Exception as ex:
            print("COULD NOT ZERO OUTPUT:", ex)

    def close(self):
        self.safe()
        self.output(False)
        self.inst.close()

    # ---- instrument-timed list sweep ----
    def _list_sweep(self, levels, delay):
        self.output(False)        # keep the setup steps off the junction
        self.w(':TRAC:CLE "defbuffer1"')

        # first: create the voltage list
        for k in range(0, len(levels), 100):
            cmd = ":SOUR:LIST:VOLT" if k == 0 else ":SOUR:LIST:VOLT:APP"
            self.w(f"{cmd} " + ",".join(f"{v:.5f}" for v in levels[k:k + 100]))

        # second: set up the sweep that uses it
        # start index, delay, count, failAbort (stop on current limit), buffer
        self.w(f':SOUR:SWE:VOLT:LIST 1, {delay:.4f}, 1, ON, "defbuffer1"')
        errs = self.check_errors()
        if errs:
            raise RuntimeError(f"Sweep setup failed: {errs}")

        # third: run it and read the data
        self.w(":INIT")
        self.q("*OPC?")                       # blocks until the sweep is done
        n = int(self.q(':TRAC:ACT? "defbuffer1"'))
        raw = self.q(f':TRAC:DATA? 1, {n}, "defbuffer1", READ, SOUR, REL').split(",")
        self.set_v(0.0)
        self.output(True)                     # the sweep ends with the output off
        a = np.array(raw, float).reshape(-1, 3)
        return pd.DataFrame({"I": a[:, 0], "V": a[:, 1], "t": a[:, 2]}), n < len(levels)

    def burst(self, V_A, M=None, t_pulse=None, t_blank=None, dt_point=None):
        M = self.m_pulses if M is None else M
        t_pulse = self.t_pulse if t_pulse is None else t_pulse
        t_blank = self.t_blank if t_blank is None else t_blank
        dt_point = self.dt_point if dt_point is None else dt_point

        n_p, n_b = round(t_pulse / dt_point), round(t_blank / dt_point)
        assert n_p >= 1 and n_b >= 1, "T_PULSE and T_BLANK must be >= DT_POINT"
        levels = ([V_A] * n_p + [0.0] * n_b + [-V_A] * n_p + [0.0] * n_b) * M
        t_meas = self.nplc_burst / 60
        delay = max(dt_point - t_meas - T_OVERHEAD, 0.0)
        self.set_range(2)
        self.w(f":SENS:CURR:NPLC {self.nplc_burst}")
        self.w(f":SENS:CURR:RANG {pick_range(self.ilim)}")
        d, aborted = self._list_sweep(levels, delay)
        pos, neg = d[d.V > V_A / 2], d[d.V < -V_A / 2]
        out = dict(I_pos=pos.I.abs().max(), I_neg=neg.I.abs().max(),
                   compliance=aborted or self.in_compliance(d.I.abs().max()),
                   t_point=float(np.median(np.diff(d.t))) if len(d) > 1 else np.nan)
        out["t_pulse_meas"] = out["t_point"] * n_p
        out["t_blank_meas"] = out["t_point"] * n_b
        if not self.timing_reported:
            print(f"  measured burst timing: {out['t_point'] * 1e3:.1f} ms/point -> pulse "
                  f"{out['t_pulse_meas'] * 1e3:.0f} ms, blank {out['t_blank_meas'] * 1e3:.0f} ms "
                  f"(asked {t_pulse * 1e3:.0f} / {t_blank * 1e3:.0f})")
            self.timing_reported = True
        return out

    def iv(self, v=None, n=None):
        """I-V sweep on the 200 mV range; R from a linear fit (intercept absorbs offsets)."""
        v = self.v_iv if v is None else v
        n = self.n_iv if n is None else n
        self.set_v(0.0)
        self.set_range(0.2)
        self.w(f":SENS:CURR:NPLC {self.nplc_iv}")
        self.w(":SENS:CURR:RANG:AUTO ON")
        d, _ = self._list_sweep(list(np.linspace(-v, v, n)), self.delay_iv)
        self.last_iv = d                       # raw sweep (I, V, t) for plotting/saving
        self.set_range(2)
        slope, icpt = np.polyfit(d.V, d.I, 1)
        resid = d.I - (slope * d.V + icpt)
        R = 1.0 / slope
        return R, float(resid.std() * R / v)   # R, relative fit residual

    def iv_fast(self, v=None, n=None):
        """I-V sweep on the 200 mV range; R from a linear fit (intercept absorbs offsets)."""
        v = self.v_iv if v is None else v
        n = self.n_iv if n is None else n
        self.set_v(0.0)
        self.set_range(0.2)
        self.w(f":SENS:CURR:NPLC {self.nplc_iv}")
        self.w(":SENS:CURR:RANG 1e-5")
        # d, _ = self._list_sweep(list(np.linspace(-v, v, n)), self.delay_iv)
        # self.last_iv = d                       # raw sweep (I, V, t) for plotting/saving
        # self.set_range(2)
        d, aborted = self._list_sweep(list(np.linspace(-v, v, n)), self.delay_iv)
        self.last_iv = d
        self.set_range(2)
        if aborted:
            raise RuntimeError("current limit during I-V (short?)")
        slope, icpt = np.polyfit(d.V, d.I, 1)
        resid = d.I - (slope * d.V + icpt)
        R = 1.0 / slope
        return R, float(resid.std() * R / v)   # R, relative fit residual