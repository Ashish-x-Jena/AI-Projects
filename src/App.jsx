import { useState } from "react";
import { FileText, Loader2, Sparkles, UploadCloud, CheckCircle2, AlertCircle, BriefcaseBusiness } from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "/api";

function ResultList({ title, items, tone = "accent" }) {
  if (!items?.length) return null;
  return (
    <section className="result-section">
      <h3>{title}</h3>
      <ul className={`result-list ${tone}`}>
        {items.map((item, index) => <li key={`${title}-${index}`}>{item}</li>)}
      </ul>
    </section>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);

  function chooseFile(nextFile) {
    setError("");
    if (!nextFile) return;
    if (nextFile.type !== "application/pdf" && !nextFile.name.toLowerCase().endsWith(".pdf")) {
      setError("Please choose a PDF resume.");
      return;
    }
    if (nextFile.size > 10 * 1024 * 1024) {
      setError("The resume must be smaller than 10 MB.");
      return;
    }
    setFile(nextFile);
    setResult(null);
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    const form = new FormData();
    form.append("resume", file);
    if (jobDescription.trim()) form.append("job_description", jobDescription.trim());
    try {
      const response = await fetch(`${API_URL}/analyze-resume`, { method: "POST", body: form });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Analysis failed. Please try again.");
      setResult(payload);
    } catch (err) {
      setError(err.message || "Could not connect to the analyzer.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="analyzer-shell">
      <header className="site-header">
        <div className="brand"><span className="brand-mark"><Sparkles size={18} /></span><span>AI Resume Analyzer</span></div>
      </header>

      <section className="hero">
        <div className="eyebrow"><Sparkles size={14} /> Built for your next opportunity</div>
        <h1>Turn your resume into your <em>strongest</em> application.</h1>
        <p className="hero-copy">Upload your resume and get clear, practical feedback on ATS compatibility, strengths, gaps, and the roles where you can stand out.</p>
      </section>

      <section className="workspace">
        <div className="upload-panel panel">
          <div className="panel-heading"><div><p className="kicker">01 / Upload</p><h2>Start with your resume</h2></div><FileText className="heading-icon" size={23} /></div>
          <div
            className={`drop-zone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => { event.preventDefault(); setDragging(false); chooseFile(event.dataTransfer.files[0]); }}
          >
            <input id="resume-file" type="file" accept="application/pdf,.pdf" onChange={(event) => chooseFile(event.target.files[0])} />
            <label htmlFor="resume-file">
              <span className="upload-icon">{file ? <CheckCircle2 size={26} /> : <UploadCloud size={26} />}</span>
              <strong>{file ? file.name : "Drop your PDF here"}</strong>
              <span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB · Ready to analyze` : "or click to browse · PDF up to 10 MB"}</span>
            </label>
          </div>
          <label className="field-label" htmlFor="job-description"><BriefcaseBusiness size={15} /> Target job description <span>optional</span></label>
          <textarea id="job-description" value={jobDescription} onChange={(event) => setJobDescription(event.target.value)} placeholder="Paste a job description to get tailored keyword and role feedback..." rows={5} />
          <button className="primary-button" onClick={analyze} disabled={!file || loading}>{loading ? <><Loader2 size={17} className="spin" /> Analyzing your resume...</> : <><Sparkles size={17} /> Analyze my resume</>}</button>
          {error && <div className="error-message"><AlertCircle size={16} /> {error}</div>}
        </div>

        <div className="results-panel">
          {!result && !loading && <div className="empty-results"><div className="empty-orbit"><Sparkles size={28} /></div><h2>Your insights will appear here</h2><p>Upload a resume to see your ATS score, standout strengths, and the highest-impact improvements.</p></div>}
          {loading && <div className="empty-results"><Loader2 size={35} className="spin accent-icon" /><h2>Reading between the lines...</h2><p>Extracting your experience and comparing it with proven hiring signals.</p></div>}
          {result && <div className="results-content">
            <div className="results-top"><div><p className="kicker">02 / Your results</p><h2>Resume intelligence</h2></div><div className="score-ring"><strong>{result.overall_score}</strong><span>/ 100</span></div></div>
            <div className="score-label">ATS compatibility score</div>
            <div className="score-bar"><span style={{ width: `${result.overall_score}%` }} /></div>
            <div className="summary-card"><p className="kicker">Profile summary</p><p>{result.profile_summary}</p></div>
            <div className="result-grid"><ResultList title="Key skills" items={result.key_skills} /><ResultList title="Recommended roles" items={result.recommended_roles} tone="green" /></div>
            <ResultList title="Strengths" items={result.strengths} tone="green" />
            <ResultList title="Areas to improve" items={result.areas_of_improvement} tone="warm" />
            <ResultList title="Actionable tips" items={result.tips_to_improve} />
            {result.experience_summary && <div className="experience-note"><p className="kicker">Experience snapshot</p><p>{result.experience_summary}</p></div>}
          </div>}
        </div>
      </section>
      <footer><span>AI Resume Analyzer</span><span>Private by design · Your resume is processed only for this analysis</span></footer>
    </main>
  );
}
