"use client";

import { useMemo, useState } from "react";
import probeData from "./qa-probe.json";

type QaScore = { exact: boolean; token_f1: number };
type QaRecord = {
  id: string;
  image: string;
  image_width: number;
  image_height: number;
  domain: string;
  question: string;
  answer: string;
  question_type: string;
  question_form: string;
  predictions: Record<string, string>;
  display_predictions: Record<string, string>;
  leaked_turn: boolean;
  scores: Record<string, QaScore>;
};
type GroundingScore = {
  parsed_point: [number, number] | null;
  parse_success: boolean;
  within_element_bounds: boolean;
  pixel_distance: number | null;
};
type GroundingRecord = {
  id: string;
  image: string;
  image_width: number;
  image_height: number;
  domain: string;
  description: string;
  answer: string;
  bbox_normalized: [number, number, number, number];
  target_point_normalized: [number, number];
  predictions: Record<string, string>;
  scores: Record<string, GroundingScore>;
};
type Metric = { exact_accuracy: number; mean_token_f1: number };
type GroundingMetric = {
  parse_rate: number;
  click_accuracy: number;
  mean_pixel_distance: number | null;
  median_pixel_distance: number | null;
  accuracy_within_25px: number;
  accuracy_within_50px: number;
  accuracy_within_100px: number;
};
type DashboardPayload = {
  meta: { license: string; checkpoint_labels: Record<string, string>; latest_label: string; latest_step: number };
  qa: {
    meta: { examples: number; unique_screenshots: number; question_types: Record<string, number>; turn_leak_examples: number };
    metrics: Record<string, Metric>;
    records: QaRecord[];
  };
  grounding: {
    meta: { examples: number; unique_screenshots: number };
    metrics: Record<string, GroundingMetric>;
    records: GroundingRecord[];
  };
};

const data = probeData as DashboardPayload;
const PAGE_SIZE = 12;
const CHECKPOINTS = ["baseline", data.meta.latest_label];
const CHECKPOINT_LABELS = data.meta.checkpoint_labels;

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function ScoreTag({ score, label }: { score: QaScore; label: string }) {
  return (
    <span className={score.exact ? "score scoreGood" : "score"}>
      {label} · {score.exact ? "exact" : `F1 ${score.token_f1.toFixed(2)}`}
    </span>
  );
}

function MetricCard({ label, value, note, tone }: { label: string; value: string; note: string; tone?: "teal" | "orange" }) {
  return (
    <div className={`metricCard ${tone ?? ""}`}>
      <span>{label}</span><strong>{value}</strong><small>{note}</small>
    </div>
  );
}

function QaCard({ record, eager }: { record: QaRecord; eager: boolean }) {
  return (
    <article className="recordCard">
      <div className="imageFrame" style={{ aspectRatio: `${record.image_width}/${record.image_height}` }}>
        <img alt={`Browser screenshot for: ${record.question}`} height={record.image_height} loading={eager ? "eager" : "lazy"} src={record.image} width={record.image_width} />
        <div className="imageMeta"><span>{record.domain}</span><span>{record.image_width} × {record.image_height}</span></div>
      </div>
      <div className="recordBody">
        <div className="recordTags"><span>{record.question_type}</span><span>{record.question_form?.replace("_", " ")}</span>{record.leaked_turn && <span className="leakTag">turn leak</span>}</div>
        <h3>{record.question}</h3>
        <div className="answerBlock reference"><p>Reference answer</p><div>{record.answer}</div></div>
        <div className="predictionGrid">
          {CHECKPOINTS.map((checkpoint) => <div className={`answerBlock ${checkpoint}`} key={checkpoint}><p>{CHECKPOINT_LABELS[checkpoint]}</p><div>{record.display_predictions[checkpoint]}</div><ScoreTag label={checkpoint === "baseline" ? "base" : "latest"} score={record.scores[checkpoint]} /></div>)}
        </div>
        <details><summary>Show raw model outputs</summary><pre>{CHECKPOINTS.map((checkpoint) => `${CHECKPOINT_LABELS[checkpoint]}\n${record.predictions[checkpoint]}`).join("\n\n")}</pre></details>
      </div>
    </article>
  );
}

function GroundingCard({ record, eager }: { record: GroundingRecord; eager: boolean }) {
  const [x1, y1, x2, y2] = record.bbox_normalized;
  const [targetX, targetY] = record.target_point_normalized;
  return (
    <article className="recordCard groundingCard">
      <div className="imageFrame groundingImage" style={{ aspectRatio: `${record.image_width}/${record.image_height}` }}>
        <img alt={`Browser screenshot grounding target: ${record.description}`} height={record.image_height} loading={eager ? "eager" : "lazy"} src={record.image} width={record.image_width} />
        <span className="targetBox" style={{ left: `${x1 / 10}%`, top: `${y1 / 10}%`, width: `${(x2 - x1) / 10}%`, height: `${(y2 - y1) / 10}%` }} />
        <span className="clickMarker targetPoint" style={{ left: `${targetX / 10}%`, top: `${targetY / 10}%` }} title="Ground truth center">GT</span>
        {CHECKPOINTS.map((checkpoint) => {
          const point = record.scores[checkpoint].parsed_point;
          return point ? <span className={`clickMarker ${checkpoint}`} key={checkpoint} style={{ left: `${point[0] / 10}%`, top: `${point[1] / 10}%` }} title={`${CHECKPOINT_LABELS[checkpoint]} prediction`}>{checkpoint === "baseline" ? "B" : "L"}</span> : null;
        })}
        <div className="imageMeta"><span>{record.domain}</span><span>{record.image_width} × {record.image_height}</span></div>
      </div>
      <div className="recordBody">
        <div className="markerLegend"><span className="legendTarget">GT target + bounds</span><span className="legendBaseline">Baseline</span><span className="legendLatest">{CHECKPOINT_LABELS[data.meta.latest_label]}</span></div>
        <p className="groundingLabel">Element description</p>
        <h3>{record.description}</h3>
        <div className="groundingScores">
          {CHECKPOINTS.map((checkpoint) => {
            const score = record.scores[checkpoint];
            return (
              <div className={score.within_element_bounds ? "groundScore hit" : "groundScore miss"} key={checkpoint}>
                <p>{CHECKPOINT_LABELS[checkpoint]}</p>
                <strong>{score.within_element_bounds ? "Inside" : "Miss"}</strong>
                <span>{score.pixel_distance === null ? "unparsed" : `${score.pixel_distance.toFixed(1)} px from center`}</span>
                <small>{score.parsed_point ? `[${score.parsed_point.map((value) => Math.round(value)).join(", ")}]` : "no point"}</small>
              </div>
            );
          })}
        </div>
        <details><summary>Show raw checkpoint outputs</summary><pre>{CHECKPOINTS.map((checkpoint) => `${CHECKPOINT_LABELS[checkpoint]}\n${record.predictions[checkpoint]}`).join("\n\n")}</pre></details>
      </div>
    </article>
  );
}

export default function Home() {
  const [task, setTask] = useState<"qa" | "grounding">("qa");
  const [query, setQuery] = useState("");
  const [type, setType] = useState("all");
  const [status, setStatus] = useState("all");
  const [visible, setVisible] = useState(PAGE_SIZE);

  const qaFiltered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return data.qa.records.filter((record) => {
      const matchesType = type === "all" || record.question_type === type;
      const matchesStatus = status === "all" || (status === "recovered" && record.scores[data.meta.latest_label].exact) || (status === "still-wrong" && !record.scores[data.meta.latest_label].exact);
      const matchesQuery = !needle || [record.question, record.answer, record.domain, record.display_predictions[data.meta.latest_label]].join(" ").toLowerCase().includes(needle);
      return matchesType && matchesStatus && matchesQuery;
    });
  }, [query, status, type]);

  const groundFiltered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return data.grounding.records.filter((record) => {
      const matchesStatus = status === "all" || (status === "hit" && record.scores[data.meta.latest_label].within_element_bounds) || (status === "miss" && !record.scores[data.meta.latest_label].within_element_bounds);
      const matchesQuery = !needle || [record.description, record.domain].join(" ").toLowerCase().includes(needle);
      return matchesStatus && matchesQuery;
    });
  }, [query, status]);

  const filtered = task === "qa" ? qaFiltered : groundFiltered;
  const switchTask = (next: "qa" | "grounding") => { setTask(next); setQuery(""); setType("all"); setStatus("all"); setVisible(PAGE_SIZE); };
  const resetFilters = () => { setQuery(""); setType("all"); setStatus("all"); setVisible(PAGE_SIZE); };

  return (
    <main>
      <header className="hero">
        <nav><div className="brandMark">SP</div><div className="brandText"><strong>Spider Lab</strong><span>EXP002 · data explorer</span></div><a href="#examples">Browse examples ↓</a></nav>
        <div className="heroGrid">
          <div className="heroCopy">
            <p className="eyebrow">MolmoWeb · fixed validation probe</p>
            <h1>See the data.<br />See every click.</h1>
            <p className="dek">A visual audit of held-out browser questions and grounding targets, with reference answers, target bounds, and checkpoint predictions side by side.</p>
            <div className="diagnosis"><span className="pulse" /><p><strong>Latest validated checkpoint:</strong> step {data.meta.latest_step}. This snapshot refreshes after each completed training-stage probe.</p></div>
          </div>
          <div className="metricGrid">
            <MetricCard label={`QA · latest (step ${data.meta.latest_step})`} value={percent(data.qa.metrics[data.meta.latest_label].exact_accuracy)} note={`token F1 ${data.qa.metrics[data.meta.latest_label].mean_token_f1.toFixed(3)}`} tone="teal" />
            <MetricCard label="QA · baseline" value={percent(data.qa.metrics.baseline.exact_accuracy)} note={`token F1 ${data.qa.metrics.baseline.mean_token_f1.toFixed(3)}`} />
            <MetricCard label={`Grounding · latest (step ${data.meta.latest_step})`} value={percent(data.grounding.metrics[data.meta.latest_label].click_accuracy)} note={`median ${data.grounding.metrics[data.meta.latest_label].median_pixel_distance?.toFixed(1)} px`} tone="teal" />
            <MetricCard label="Grounding · baseline" value={percent(data.grounding.metrics.baseline.click_accuracy)} note={`median ${data.grounding.metrics.baseline.median_pixel_distance?.toFixed(1)} px`} tone="orange" />
          </div>
        </div>
      </header>

      <section className="taskTabs" aria-label="Dataset task">
        <button className={task === "qa" ? "active" : ""} onClick={() => switchTask("qa")}><span>01</span> Screenshot QA <b>{data.qa.meta.examples}</b></button>
        <button className={task === "grounding" ? "active" : ""} onClick={() => switchTask("grounding")}><span>02</span> GUI grounding <b>{data.grounding.meta.examples}</b></button>
      </section>

      <section className="explorer" id="examples">
        <aside className="filters">
          <div><p className="filterLabel">Search</p><input aria-label="Search dataset examples" onChange={(event) => { setQuery(event.target.value); setVisible(PAGE_SIZE); }} placeholder={task === "qa" ? "button, price, headline…" : "settings, link, menu…"} type="search" value={query} /></div>
          {task === "qa" && <div><p className="filterLabel">Question type</p><div className="filterStack"><button className={type === "all" ? "active" : ""} onClick={() => setType("all")}><span>All examples</span><b>{data.qa.meta.examples}</b></button>{Object.entries(data.qa.meta.question_types).map(([name, count]) => <button className={type === name ? "active" : ""} key={name} onClick={() => { setType(name); setVisible(PAGE_SIZE); }}><span>{name}</span><b>{count}</b></button>)}</div></div>}
          <div><p className="filterLabel">{task === "qa" ? "Latest result" : "Latest click"}</p><div className="filterStack"><button className={status === "all" ? "active" : ""} onClick={() => setStatus("all")}><span>Any result</span></button>{task === "qa" ? <><button className={status === "recovered" ? "active" : ""} onClick={() => setStatus("recovered")}><span>Exact answer</span></button><button className={status === "still-wrong" ? "active" : ""} onClick={() => setStatus("still-wrong")}><span>Needs review</span></button></> : <><button className={status === "hit" ? "active" : ""} onClick={() => setStatus("hit")}><span>Inside target</span></button><button className={status === "miss" ? "active" : ""} onClick={() => setStatus("miss")}><span>Missed target</span></button></>}</div></div>
          <div className="datasetNote"><strong>{task === "qa" ? "What one QA record contains" : "How to read the overlay"}</strong><p>{task === "qa" ? "Screenshot, natural-language question, concise reference answer, QA type, form, domain, and URL." : `Green marks the annotated element bounds and center. Gray is baseline; teal is the latest model click at step ${data.meta.latest_step}.`}</p><small>{data.meta.license}</small></div>
        </aside>

        <div className="results">
          <div className="resultsHeader"><div><p className="eyebrow">{task === "qa" ? "QA records" : "Grounding records"}</p><h2>{filtered.length} matching examples</h2></div>{(query || type !== "all" || status !== "all") && <button className="reset" onClick={resetFilters}>Reset filters</button>}</div>
          <div className="cards">{task === "qa" ? qaFiltered.slice(0, visible).map((record, index) => <QaCard eager={index < 4} key={record.id} record={record} />) : groundFiltered.slice(0, visible).map((record, index) => <GroundingCard eager={index < 4} key={record.id} record={record} />)}</div>
          {visible < filtered.length && <button className="loadMore" onClick={() => setVisible((value) => value + PAGE_SIZE)}>Load {Math.min(PAGE_SIZE, filtered.length - visible)} more</button>}
        </div>
      </section>
      <footer><span>Spider EXP002 · generated from immutable Kaggle probe artifacts</span><span>MolmoWeb · ODC-BY 1.0</span></footer>
    </main>
  );
}
