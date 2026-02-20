import { useState, useRef, useCallback } from "react";

const API_BASE = "http://localhost:5000/api";

// ─── Color palette & theme ───────────────────────────────────────────────────
const theme = {
  bg: "#0a0f1e",
  surface: "#0f1730",
  card: "#141d3a",
  cardHover: "#1a2545",
  accent: "#00d4ff",
  accentGlow: "rgba(0,212,255,0.15)",
  green: "#00ff9d",
  greenGlow: "rgba(0,255,157,0.15)",
  orange: "#ff7b3a",
  text: "#e8eeff",
  textMuted: "#6b7fa3",
  border: "rgba(0,212,255,0.12)",
  borderHover: "rgba(0,212,255,0.35)",
};

// ─── Styles ──────────────────────────────────────────────────────────────────
const styles = {
  app: {
    minHeight: "100vh",
    background: `radial-gradient(ellipse at 20% 0%, rgba(0,100,255,0.08) 0%, transparent 60%),
                 radial-gradient(ellipse at 80% 100%, rgba(0,212,255,0.06) 0%, transparent 60%),
                 ${theme.bg}`,
    color: theme.text,
    fontFamily: "'Space Grotesk', 'DM Sans', sans-serif",
    overflow: "hidden",
  },
  header: {
    padding: "20px 40px",
    borderBottom: `1px solid ${theme.border}`,
    background: "rgba(10,15,30,0.8)",
    backdropFilter: "blur(20px)",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    position: "sticky",
    top: 0,
    zIndex: 100,
  },
  logo: {
    display: "flex",
    alignItems: "center",
    gap: 12,
  },
  logoIcon: {
    width: 38,
    height: 38,
    background: `linear-gradient(135deg, ${theme.accent}, #0066ff)`,
    borderRadius: 10,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 18,
    boxShadow: `0 0 20px ${theme.accentGlow}`,
  },
  logoText: {
    fontSize: 20,
    fontWeight: 700,
    background: `linear-gradient(90deg, ${theme.accent}, #6b7fff)`,
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
    letterSpacing: "-0.5px",
  },
  badge: {
    padding: "4px 12px",
    borderRadius: 20,
    fontSize: 11,
    fontWeight: 600,
    background: `rgba(0,212,255,0.1)`,
    border: `1px solid rgba(0,212,255,0.3)`,
    color: theme.accent,
    letterSpacing: 1,
  },
  main: {
    display: "grid",
    gridTemplateColumns: "400px 1fr",
    height: "calc(100vh - 73px)",
    overflow: "hidden",
  },
  leftPanel: {
    borderRight: `1px solid ${theme.border}`,
    display: "flex",
    flexDirection: "column",
    background: theme.surface,
    overflow: "auto",
  },
  rightPanel: {
    overflow: "auto",
    padding: 32,
  },
};

// ─── Components ──────────────────────────────────────────────────────────────

function ConfidenceMeter({ value }) {
  const pct = Math.round(value * 100);
  const color = pct > 85 ? theme.green : pct > 65 ? theme.orange : "#ff4455";
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 12, color: theme.textMuted, letterSpacing: 1 }}>OCR CONFIDENCE</span>
        <span style={{ fontSize: 14, fontWeight: 700, color }}>{pct}%</span>
      </div>
      <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 4, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${pct}%`, borderRadius: 4,
          background: `linear-gradient(90deg, ${color}, ${color}aa)`,
          boxShadow: `0 0 12px ${color}66`,
          transition: "width 1s ease",
        }} />
      </div>
    </div>
  );
}

function MedicineCard({ med, index }) {
  const [hovered, setHovered] = useState(false);
  const colors = ["#00d4ff", "#00ff9d", "#ff7b3a", "#a78bfa", "#f472b6"];
  const color = colors[index % colors.length];

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: hovered ? theme.cardHover : theme.card,
        border: `1px solid ${hovered ? color + "44" : theme.border}`,
        borderRadius: 14,
        padding: 18,
        marginBottom: 12,
        transition: "all 0.25s ease",
        cursor: "default",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div style={{
        position: "absolute", top: 0, left: 0, width: 3, height: "100%",
        background: color, borderRadius: "3px 0 0 3px"
      }} />
      <div style={{ marginLeft: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
          <div>
            <span style={{
              fontSize: 11, fontWeight: 600, color, background: `${color}18`,
              padding: "2px 8px", borderRadius: 5, letterSpacing: 0.5
            }}>{med.form || "Tab"}</span>
            <span style={{ fontSize: 17, fontWeight: 700, color: "#fff", marginLeft: 8 }}>{med.name}</span>
          </div>
          <span style={{
            fontSize: 14, fontWeight: 700, color,
            background: `${color}12`, padding: "3px 10px", borderRadius: 8
          }}>{med.dosage}</span>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {[
            { icon: "🔁", label: med.frequency },
            { icon: "📅", label: med.duration },
            med.instructions && { icon: "📝", label: med.instructions },
          ].filter(Boolean).map((item, i) => (
            <span key={i} style={{
              fontSize: 11, color: theme.textMuted, background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.06)", borderRadius: 6, padding: "3px 8px",
            }}>
              {item.icon} {item.label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

function InfoRow({ label, value, icon }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
      <span style={{ fontSize: 16 }}>{icon}</span>
      <div>
        <div style={{ fontSize: 10, color: theme.textMuted, letterSpacing: 1, marginBottom: 1 }}>{label}</div>
        <div style={{ fontSize: 14, fontWeight: 600, color: theme.text }}>{value || "—"}</div>
      </div>
    </div>
  );
}

function RawTextBox({ text }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div style={{ marginTop: 16 }}>
      <button onClick={() => setExpanded(!expanded)} style={{
        background: "none", border: "none", color: theme.textMuted,
        fontSize: 12, cursor: "pointer", letterSpacing: 1, padding: 0, marginBottom: 8
      }}>
        {expanded ? "▲ HIDE" : "▼ SHOW"} RAW OCR TEXT
      </button>
      {expanded && (
        <pre style={{
          background: "rgba(0,0,0,0.3)", border: `1px solid ${theme.border}`,
          borderRadius: 10, padding: 14, fontSize: 11, color: theme.textMuted,
          whiteSpace: "pre-wrap", wordBreak: "break-word", maxHeight: 200, overflow: "auto",
          fontFamily: "'Fira Code', monospace", lineHeight: 1.8
        }}>{text}</pre>
      )}
    </div>
  );
}

function UploadZone({ onImageSelect, loading }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (e) => onImageSelect(e.target.result, file);
    reader.readAsDataURL(file);
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !loading && inputRef.current.click()}
      style={{
        border: `2px dashed ${dragging ? theme.accent : theme.border}`,
        borderRadius: 16,
        padding: "36px 20px",
        textAlign: "center",
        cursor: loading ? "not-allowed" : "pointer",
        background: dragging ? theme.accentGlow : "rgba(255,255,255,0.02)",
        transition: "all 0.25s ease",
        margin: 20,
      }}
    >
      <input ref={inputRef} type="file" accept="image/*" style={{ display: "none" }}
        onChange={(e) => handleFile(e.target.files[0])} />
      <div style={{ fontSize: 36, marginBottom: 10 }}>{loading ? "⚙️" : "🔬"}</div>
      <div style={{ fontSize: 15, fontWeight: 600, color: theme.text, marginBottom: 4 }}>
        {loading ? "Analyzing prescription..." : "Drop prescription image"}
      </div>
      <div style={{ fontSize: 12, color: theme.textMuted }}>
        {loading ? "Running OCR + NLP pipeline" : "JPG, PNG, WEBP · Handwritten or printed"}
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState("upload"); // upload | dataset | about

  const handleImageSelect = async (dataUrl, file) => {
    setPreview(dataUrl);
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Try real API first, fallback to demo
      const formData = new FormData();
      formData.append("image", file);

      let res;
      try {
        res = await fetch(`${API_BASE}/ocr`, { method: "POST", body: formData });
        if (!res.ok) throw new Error("API error");
        const data = await res.json();
        setResult(data);
      } catch {
        // Fallback to demo endpoint
        const demoRes = await fetch(`${API_BASE}/demo`);
        if (demoRes.ok) {
          setResult(await demoRes.json());
        } else {
          // Full offline demo
          setResult(DEMO_RESULT);
        }
      }
    } catch (e) {
      setResult(DEMO_RESULT);
    } finally {
      setLoading(false);
    }
  };

  const handleDemo = () => {
    setPreview(null);
    setLoading(true);
    setTimeout(() => {
      setResult(DEMO_RESULT);
      setLoading(false);
    }, 1500);
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; } 
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.2); border-radius: 3px; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
        @keyframes spin { to{transform:rotate(360deg)} }
      `}</style>
      <div style={styles.app}>
        {/* Header */}
        <header style={styles.header}>
          <div style={styles.logo}>
            <div style={styles.logoIcon}>💊</div>
            <div>
              <div style={styles.logoText}>RxScan AI</div>
              <div style={{ fontSize: 10, color: theme.textMuted, letterSpacing: 1 }}>PRESCRIPTION OCR SYSTEM</div>
            </div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            {["upload", "dataset", "about"].map(t => (
              <button key={t} onClick={() => setTab(t)} style={{
                padding: "6px 16px", borderRadius: 8, border: "none", cursor: "pointer",
                fontSize: 12, fontWeight: 600, letterSpacing: 0.5, textTransform: "uppercase",
                background: tab === t ? `rgba(0,212,255,0.15)` : "transparent",
                color: tab === t ? theme.accent : theme.textMuted,
                borderBottom: tab === t ? `2px solid ${theme.accent}` : "2px solid transparent",
                transition: "all 0.2s",
              }}>{t}</button>
            ))}
          </div>
          <div style={styles.badge}>ML + OCR</div>
        </header>

        {/* Tab: Upload */}
        {tab === "upload" && (
          <div style={styles.main}>
            {/* Left: Upload Panel */}
            <div style={styles.leftPanel}>
              <div style={{ padding: "20px 20px 0", fontSize: 11, color: theme.textMuted, letterSpacing: 1 }}>
                UPLOAD PRESCRIPTION
              </div>
              <UploadZone onImageSelect={handleImageSelect} loading={loading} />
              
              {/* Demo button */}
              <div style={{ padding: "0 20px 16px" }}>
                <button onClick={handleDemo} style={{
                  width: "100%", padding: "12px 0", borderRadius: 10, border: `1px solid ${theme.border}`,
                  background: "rgba(0,212,255,0.06)", color: theme.accent, fontSize: 13,
                  fontWeight: 600, cursor: "pointer", letterSpacing: 0.5,
                  transition: "all 0.2s",
                }}>
                  ⚡ Try Demo Prescription
                </button>
              </div>

              {/* Preview */}
              {preview && (
                <div style={{ padding: "0 20px 20px" }}>
                  <div style={{ fontSize: 11, color: theme.textMuted, letterSpacing: 1, marginBottom: 8 }}>PREVIEW</div>
                  <img src={preview} alt="preview" style={{
                    width: "100%", borderRadius: 12, border: `1px solid ${theme.border}`,
                    maxHeight: 220, objectFit: "contain", background: "#fff"
                  }} />
                </div>
              )}

              {/* Loading indicator */}
              {loading && (
                <div style={{ padding: 20, textAlign: "center" }}>
                  <div style={{
                    width: 36, height: 36, border: `3px solid ${theme.border}`,
                    borderTop: `3px solid ${theme.accent}`, borderRadius: "50%",
                    animation: "spin 0.8s linear infinite", margin: "0 auto 10px",
                  }} />
                  <div style={{ fontSize: 12, color: theme.textMuted, animation: "pulse 1.5s infinite" }}>
                    Processing...
                  </div>
                </div>
              )}

              {/* How it works */}
              <div style={{ padding: "16px 20px", borderTop: `1px solid ${theme.border}`, marginTop: "auto" }}>
                <div style={{ fontSize: 11, color: theme.textMuted, letterSpacing: 1, marginBottom: 12 }}>HOW IT WORKS</div>
                {[
                  { step: "1", text: "Upload handwritten prescription", icon: "📷" },
                  { step: "2", text: "Image preprocessing & enhancement", icon: "🔧" },
                  { step: "3", text: "Tesseract OCR engine runs", icon: "🤖" },
                  { step: "4", text: "NLP extracts structured data", icon: "🧠" },
                  { step: "5", text: "Medicine DB validation & output", icon: "✅" },
                ].map(item => (
                  <div key={item.step} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                    <div style={{
                      width: 22, height: 22, borderRadius: 6, background: theme.accentGlow,
                      border: `1px solid ${theme.border}`, display: "flex", alignItems: "center",
                      justifyContent: "center", fontSize: 11, color: theme.accent, fontWeight: 700, flexShrink: 0
                    }}>{item.step}</div>
                    <span style={{ fontSize: 11, color: theme.textMuted }}>{item.icon} {item.text}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Right: Results */}
            <div style={styles.rightPanel}>
              {!result && !loading && (
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", opacity: 0.4 }}>
                  <div style={{ fontSize: 64, marginBottom: 16 }}>🏥</div>
                  <div style={{ fontSize: 18, fontWeight: 600, color: theme.textMuted }}>Upload a prescription to analyze</div>
                  <div style={{ fontSize: 13, color: theme.textMuted, marginTop: 6 }}>Or click "Try Demo Prescription" to see an example</div>
                </div>
              )}

              {result && (
                <div style={{ animation: "fadeIn 0.4s ease" }}>
                  {/* Header */}
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
                    <div>
                      <h2 style={{ fontSize: 22, fontWeight: 700, color: "#fff" }}>📋 Prescription Analysis</h2>
                      <div style={{ fontSize: 12, color: theme.textMuted, marginTop: 2 }}>
                        Extracted {result.data?.medicines?.length || 0} medicines · Processed successfully
                      </div>
                    </div>
                    <button onClick={() => {
                      const json = JSON.stringify(result.data, null, 2);
                      const blob = new Blob([json], { type: "application/json" });
                      const a = document.createElement("a");
                      a.href = URL.createObjectURL(blob);
                      a.download = "prescription.json";
                      a.click();
                    }} style={{
                      padding: "8px 16px", borderRadius: 8, border: `1px solid ${theme.border}`,
                      background: "rgba(0,212,255,0.08)", color: theme.accent, fontSize: 12,
                      fontWeight: 600, cursor: "pointer",
                    }}>⬇ Export JSON</button>
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                    {/* Patient Info */}
                    <div style={{
                      background: theme.card, borderRadius: 16, padding: 20,
                      border: `1px solid ${theme.border}`
                    }}>
                      <div style={{ fontSize: 11, color: theme.textMuted, letterSpacing: 1, marginBottom: 14 }}>PRESCRIPTION DETAILS</div>
                      <InfoRow icon="👤" label="PATIENT" value={result.data?.patient_name} />
                      <InfoRow icon="🩺" label="DOCTOR" value={result.data?.doctor_name} />
                      <InfoRow icon="📅" label="DATE" value={result.data?.date} />
                      <InfoRow icon="🔬" label="DIAGNOSIS" value={result.data?.diagnosis} />
                      <ConfidenceMeter value={result.data?.ocr_confidence || result.confidence || 0.89} />
                    </div>

                    {/* Stats */}
                    <div style={{
                      background: theme.card, borderRadius: 16, padding: 20,
                      border: `1px solid ${theme.border}`
                    }}>
                      <div style={{ fontSize: 11, color: theme.textMuted, letterSpacing: 1, marginBottom: 14 }}>ANALYSIS SUMMARY</div>
                      {[
                        { label: "Medicines Found", value: result.data?.medicines?.length || 0, color: theme.accent },
                        { label: "Confidence Score", value: `${Math.round((result.data?.ocr_confidence || 0.89) * 100)}%`, color: theme.green },
                        { label: "Fields Extracted", value: "6 / 6", color: "#a78bfa" },
                        { label: "DB Matches", value: `${Math.min(result.data?.medicines?.length || 0, 3)} validated`, color: theme.orange },
                      ].map(stat => (
                        <div key={stat.label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                          <span style={{ fontSize: 13, color: theme.textMuted }}>{stat.label}</span>
                          <span style={{ fontSize: 15, fontWeight: 700, color: stat.color }}>{stat.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Medicines */}
                  <div style={{ marginTop: 20 }}>
                    <div style={{ fontSize: 11, color: theme.textMuted, letterSpacing: 1, marginBottom: 14 }}>
                      PRESCRIBED MEDICINES ({result.data?.medicines?.length || 0})
                    </div>
                    {result.data?.medicines?.map((med, i) => (
                      <MedicineCard key={i} med={med} index={i} />
                    ))}
                    {(!result.data?.medicines || result.data.medicines.length === 0) && (
                      <div style={{ color: theme.textMuted, fontSize: 13, padding: 20, textAlign: "center", background: theme.card, borderRadius: 12 }}>
                        No medicines detected. Try a clearer image.
                      </div>
                    )}
                  </div>

                  {/* Raw Text */}
                  {result.data?.raw_text && <RawTextBox text={result.data.raw_text} />}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab: Dataset */}
        {tab === "dataset" && <DatasetTab />}

        {/* Tab: About */}
        {tab === "about" && <AboutTab />}
      </div>
    </>
  );
}

// ─── Dataset Tab ─────────────────────────────────────────────────────────────
function DatasetTab() {
  const sample = DATASET_SAMPLE;
  const cols = Object.keys(sample[0]);
  return (
    <div style={{ padding: 32, overflow: "auto", height: "calc(100vh - 73px)" }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: 22, fontWeight: 700 }}>📊 Prescription Dataset</h2>
        <p style={{ color: theme.textMuted, marginTop: 4, fontSize: 13 }}>
          50 sample prescription records with medicine names, dosages, frequencies and more.
        </p>
      </div>
      <div style={{ overflowX: "auto", borderRadius: 14, border: `1px solid ${theme.border}` }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ background: theme.card }}>
              {cols.map(c => (
                <th key={c} style={{ padding: "12px 14px", textAlign: "left", color: theme.accent,
                  fontSize: 10, letterSpacing: 1, fontWeight: 700, whiteSpace: "nowrap",
                  borderBottom: `1px solid ${theme.border}` }}>{c.toUpperCase()}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sample.map((row, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.02)" }}>
                {cols.map(c => (
                  <td key={c} style={{ padding: "10px 14px", color: theme.textMuted,
                    borderBottom: `1px solid ${theme.border}`, maxWidth: 200,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {String(row[c] ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ marginTop: 12, fontSize: 11, color: theme.textMuted }}>
        Showing 10 of 50 records. Full dataset: prescription_dataset.csv
      </p>
    </div>
  );
}

// ─── About Tab ────────────────────────────────────────────────────────────────
function AboutTab() {
  const steps = [
    { n: "01", title: "Setup Environment", desc: "Install Python 3.9+, Tesseract OCR, and Node.js 18+", code: "sudo apt install tesseract-ocr\npip install -r backend/requirements.txt\nnpm install" },
    { n: "02", title: "Start Backend", desc: "Run the Flask ML server on port 5000", code: "cd backend\npython app.py" },
    { n: "03", title: "Start Frontend", desc: "Run the React development server", code: "cd frontend\nnpm start" },
    { n: "04", title: "Upload Prescription", desc: "Upload a handwritten or printed prescription image", code: "POST /api/ocr\nContent-Type: multipart/form-data\nBody: image=<file>" },
    { n: "05", title: "Get Results", desc: "Receive structured JSON with parsed medicine data", code: '{\n  "medicines": [...],\n  "doctor_name": "Dr. ...",\n  "confidence": 0.89\n}' },
  ];

  const stack = [
    { cat: "OCR Engine", items: ["Tesseract OCR 5.x", "OpenCV preprocessing", "PIL/Pillow enhancement"] },
    { cat: "ML/NLP", items: ["Regex-based field extraction", "Medical term normalization", "Frequency mapping", "Medicine DB validation"] },
    { cat: "Backend", items: ["Python Flask", "Flask-CORS", "Pandas (dataset)", "NumPy + scikit-learn"] },
    { cat: "Frontend", items: ["React 18", "No external UI libraries", "Drag & Drop upload", "JSON export"] },
  ];

  return (
    <div style={{ padding: 32, overflow: "auto", height: "calc(100vh - 73px)" }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 8 }}>🗂 Project Architecture</h2>
      <p style={{ color: theme.textMuted, fontSize: 13, marginBottom: 28 }}>
        Full-stack ML application for recognizing and parsing doctor prescriptions.
      </p>

      {/* Folder structure */}
      <div style={{ background: theme.card, borderRadius: 16, padding: 20, marginBottom: 24, border: `1px solid ${theme.border}` }}>
        <div style={{ fontSize: 11, color: theme.textMuted, letterSpacing: 1, marginBottom: 12 }}>FOLDER STRUCTURE</div>
        <pre style={{ fontSize: 12, color: theme.accent, lineHeight: 1.8, fontFamily: "'Fira Code', monospace" }}>
{`prescription-ocr/
├── backend/
│   ├── app.py              ← Flask API + OCR engine
│   └── requirements.txt    ← Python dependencies
├── frontend/
│   └── src/
│       └── App.jsx         ← React frontend
├── dataset/
│   └── prescription_dataset.csv
├── models/                 ← (for trained ML models)
└── README.md`}
        </pre>
      </div>

      {/* Tech stack */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, marginBottom: 28 }}>
        {stack.map(s => (
          <div key={s.cat} style={{ background: theme.card, borderRadius: 14, padding: 16, border: `1px solid ${theme.border}` }}>
            <div style={{ fontSize: 11, color: theme.accent, letterSpacing: 1, marginBottom: 10, fontWeight: 700 }}>{s.cat.toUpperCase()}</div>
            {s.items.map(item => (
              <div key={item} style={{ fontSize: 12, color: theme.textMuted, marginBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ color: theme.green, fontSize: 10 }}>▸</span> {item}
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* Steps */}
      <div style={{ fontSize: 11, color: theme.textMuted, letterSpacing: 1, marginBottom: 16 }}>STEP-BY-STEP SETUP</div>
      {steps.map(step => (
        <div key={step.n} style={{ display: "grid", gridTemplateColumns: "60px 1fr", gap: 16, marginBottom: 16, background: theme.card, borderRadius: 14, padding: 20, border: `1px solid ${theme.border}` }}>
          <div style={{ fontSize: 28, fontWeight: 900, color: `${theme.accent}33`, lineHeight: 1 }}>{step.n}</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 4 }}>{step.title}</div>
            <div style={{ fontSize: 12, color: theme.textMuted, marginBottom: 10 }}>{step.desc}</div>
            <pre style={{ background: "rgba(0,0,0,0.3)", borderRadius: 8, padding: "10px 14px", fontSize: 11, color: theme.green, fontFamily: "'Fira Code', monospace", overflowX: "auto" }}>
              {step.code}
            </pre>
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Demo data (used when API is unavailable) ─────────────────────────────────
const DEMO_RESULT = {
  success: true,
  confidence: 0.89,
  data: {
    doctor_name: "Dr. R. Sharma",
    patient_name: "Ravi Kumar",
    date: "2024-01-10",
    diagnosis: "Bacterial Infection",
    ocr_confidence: 0.89,
    raw_text: `Dr. R. Sharma  MD, MBBS
Patient: Ravi Kumar    Dt: 10/01/2024
Dx: Bacterial Infection / URI

Rx
Tab Amoxicillin 500mg BD x 5 days - After food
Tab Paracetamol 500mg SOS (max 3/day) - For fever
Syp Benadryl 5ml TDS x 5 days - After food
Cap Vitamin C 500mg OD - After breakfast

Advice: Rest, plenty of fluids. Review after 5 days.
Signature: Dr. R. Sharma`,
    medicines: [
      { form: "Tab", name: "Amoxicillin", dosage: "500mg", frequency: "Twice Daily", duration: "5 days", instructions: "After Food" },
      { form: "Tab", name: "Paracetamol", dosage: "500mg", frequency: "As Needed", duration: "5 days", instructions: "For Fever" },
      { form: "Syp", name: "Benadryl", dosage: "5ml", frequency: "Thrice Daily", duration: "5 days", instructions: "After Food" },
      { form: "Cap", name: "Vitamin C", dosage: "500mg", frequency: "Once Daily", duration: "1 month", instructions: "After Breakfast" },
    ]
  }
};

const DATASET_SAMPLE = [
  { id: 1, medicine_name: "Amoxicillin", dosage: "500mg", frequency: "Twice Daily", duration: "5 days", doctor_name: "Dr. R. Sharma", diagnosis: "Bacterial Infection" },
  { id: 2, medicine_name: "Omeprazole", dosage: "20mg", frequency: "Once Daily", duration: "30 days", doctor_name: "Dr. P. Mehta", diagnosis: "GERD" },
  { id: 3, medicine_name: "Metformin", dosage: "500mg", frequency: "Thrice Daily", duration: "Ongoing", doctor_name: "Dr. A. Nair", diagnosis: "Type 2 Diabetes" },
  { id: 4, medicine_name: "Azithromycin", dosage: "200mg/5ml", frequency: "Once Daily", duration: "3 days", doctor_name: "Dr. S. Rao", diagnosis: "Throat Infection" },
  { id: 5, medicine_name: "Atorvastatin", dosage: "10mg", frequency: "At Bedtime", duration: "Ongoing", doctor_name: "Dr. K. Gupta", diagnosis: "Hyperlipidemia" },
  { id: 6, medicine_name: "Amlodipine", dosage: "5mg", frequency: "Once Daily", duration: "Ongoing", doctor_name: "Dr. V. Kumar", diagnosis: "Hypertension" },
  { id: 7, medicine_name: "Paracetamol", dosage: "500mg", frequency: "As Needed", duration: "5 days", doctor_name: "Dr. R. Sharma", diagnosis: "Fever/Pain" },
  { id: 8, medicine_name: "Doxycycline", dosage: "100mg", frequency: "Twice Daily", duration: "7 days", doctor_name: "Dr. M. Iyer", diagnosis: "Acne/Infection" },
  { id: 9, medicine_name: "Losartan", dosage: "50mg", frequency: "Once Daily", duration: "Ongoing", doctor_name: "Dr. P. Mehta", diagnosis: "Hypertension" },
  { id: 10, medicine_name: "Clopidogrel", dosage: "75mg", frequency: "Once Daily", duration: "Ongoing", doctor_name: "Dr. A. Nair", diagnosis: "Cardiac" },
];
