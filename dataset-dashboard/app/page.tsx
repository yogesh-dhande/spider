"use client";

import { useMemo, useState } from "react";
import probeData from "./qa-probe.json";
import { TrainingDataView } from "./training-data";

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
type ActionScore = {
  parse_valid: boolean;
  name_correct: boolean;
  arguments_correct: boolean;
  click_inside_bbox: boolean | null;
  click_distance_px: number | null;
  parsed_point: [number, number] | null;
};
type ActionRecord = {
  id: string;
  image: string;
  image_width: number;
  image_height: number;
  domain: string;
  instruction: string;
  source: string;
  step_index: number;
  target_action: Record<string, string | number>;
  bbox_normalized: [number, number, number, number] | null;
  predictions: Record<string, string>;
  scores: Record<string, ActionScore>;
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
type ActionMetric = {
  examples: number;
  json_parse_rate: number;
  action_name_accuracy: number;
  action_argument_accuracy: number;
  click_inside_bbox_accuracy: number | null;
  click_median_distance_px: number | null;
};
type DashboardPayload = {
  meta: { license: string; checkpoint_labels: Record<string, string>; latest_label: string; latest_step: number };
  qa: {
    meta: { split: string; examples: number; display_examples?: number; unique_screenshots: number; question_types: Record<string, number>; turn_leak_examples: number };
    metrics: Record<string, Metric>;
    records: QaRecord[];
  };
  grounding: {
    meta: { split: string; examples: number; display_examples?: number; unique_screenshots: number };
    metrics: Record<string, GroundingMetric>;
    records: GroundingRecord[];
  };
  action?: {
    meta: { split: string; scored_examples: number; display_examples: number; unique_screenshots: number; target_action_counts: Record<string, number> };
    metrics: Record<string, ActionMetric>;
    records: ActionRecord[];
  };
};

const data = probeData as DashboardPayload;
const PAGE_SIZE = 12;
const CHECKPOINTS = ["baseline", data.meta.latest_label];
const CHECKPOINT_LABELS = data.meta.checkpoint_labels;
const IS_SEALED = data.qa.meta.split === "test";
const SPLIT_LABEL = IS_SEALED ? "sealed test" : "fixed development probes";

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

function ActionCard({ record, eager }: { record: ActionRecord; eager: boolean }) {
  const bbox = record.bbox_normalized;
  const isClick = record.target_action.name === "click";
  const targetX = Number(record.target_action.x);
  const targetY = Number(record.target_action.y);
  return (
    <article className="recordCard groundingCard">
      <div className="imageFrame groundingImage" style={{ aspectRatio: `${record.image_width}/${record.image_height}` }}>
        <img alt={`Browser action screenshot for: ${record.instruction}`} height={record.image_height} loading={eager ? "eager" : "lazy"} src={record.image} width={record.image_width} />
        {bbox && <span className="targetBox" style={{ left: `${bbox[0] * 100}%`, top: `${bbox[1] * 100}%`, width: `${(bbox[2] - bbox[0]) * 100}%`, height: `${(bbox[3] - bbox[1]) * 100}%` }} />}
        {isClick && Number.isFinite(targetX) && Number.isFinite(targetY) && <span className="clickMarker targetPoint" style={{ left: `${targetX}%`, top: `${targetY}%` }} title="Ground-truth action point">GT</span>}
        {CHECKPOINTS.map((checkpoint) => {
          const point = record.scores[checkpoint].parsed_point;
          return point ? <span className={`clickMarker ${checkpoint}`} key={checkpoint} style={{ left: `${point[0]}%`, top: `${point[1]}%` }} title={`${CHECKPOINT_LABELS[checkpoint]} action point`}>{checkpoint === "baseline" ? "B" : "L"}</span> : null;
        })}
        <div className="imageMeta"><span>{record.domain}</span><span>{record.image_width} × {record.image_height}</span></div>
      </div>
      <div className="recordBody">
        <div className="markerLegend"><span className="legendTarget">GT target + bounds</span><span className="legendBaseline">{CHECKPOINT_LABELS.baseline}</span><span className="legendLatest">{CHECKPOINT_LABELS[data.meta.latest_label]}</span></div>
        <div className="recordTags"><span>{record.target_action.name}</span><span>{record.source}</span><span>step {record.step_index}</span></div>
        <h3>{record.instruction}</h3>
        <div className="answerBlock reference"><p>Target action</p><div>{JSON.stringify(record.target_action)}</div></div>
        <div className="groundingScores">
          {CHECKPOINTS.map((checkpoint) => {
            const score = record.scores[checkpoint];
            const correct = isClick ? score.click_inside_bbox === true : score.arguments_correct;
            return <div className={correct ? "groundScore hit" : "groundScore miss"} key={checkpoint}><p>{CHECKPOINT_LABELS[checkpoint]}</p><strong>{correct ? "Correct" : "Miss"}</strong><span>{score.parse_valid ? (score.name_correct ? "action name correct" : "wrong action name") : "invalid JSON"}</span><small>{score.click_distance_px == null ? (score.arguments_correct ? "arguments correct" : "arguments differ") : `${score.click_distance_px.toFixed(1)} px from target`}</small></div>;
          })}
        </div>
        <details><summary>Show raw model outputs</summary><pre>{CHECKPOINTS.map((checkpoint) => `${CHECKPOINT_LABELS[checkpoint]}\n${record.predictions[checkpoint]}`).join("\n\n")}</pre></details>
      </div>
    </article>
  );
}

export default function Home() {
  const [view, setView] = useState<"data" | "diagnostics">("data");
  const [task, setTask] = useState<"qa" | "grounding" | "action">("qa");
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

  const actionFiltered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return (data.action?.records ?? []).filter((record) => {
      const score = record.scores[data.meta.latest_label];
      const correct = record.target_action.name === "click" ? score.click_inside_bbox === true : score.arguments_correct;
      const matchesType = type === "all" || record.target_action.name === type;
      const matchesStatus = status === "all" || (status === "hit" && correct) || (status === "miss" && !correct);
      const matchesQuery = !needle || [record.instruction, record.domain, record.source, record.target_action.name].join(" ").toLowerCase().includes(needle);
      return matchesType && matchesStatus && matchesQuery;
    });
  }, [query, status, type]);

  const filtered = task === "qa" ? qaFiltered : task === "grounding" ? groundFiltered : actionFiltered;
  const switchTask = (next: "qa" | "grounding" | "action") => { setTask(next); setQuery(""); setType("all"); setStatus("all"); setVisible(PAGE_SIZE); };
  const resetFilters = () => { setQuery(""); setType("all"); setStatus("all"); setVisible(PAGE_SIZE); };

  if (view === "data") return <TrainingDataView onDiagnostics={() => setView("diagnostics")} />;

  return (
    <main>
      <header className="hero">
        <nav><div className="brandMark">SP</div><div className="brandText"><strong>Spider Lab</strong><span>EXP005 · scaling diagnostics</span></div><div className="viewSwitch"><button onClick={() => setView("data")}>Training data</button><button className="active">Model diagnostics</button></div></nav>
        <div className="heroGrid">
          <div className="heroCopy">
            <p className="eyebrow">MolmoWeb · {SPLIT_LABEL}</p>
            <h1>See the data.<br />See every click.</h1>
            <p className="dek">A visual audit of held-out browser questions and grounding targets, with reference answers, target bounds, and checkpoint predictions side by side.</p>
            <div className="diagnosis"><span className="pulse" /><p><strong>{IS_SEALED ? "Selected checkpoint" : "Latest validated checkpoint"}:</strong> step {data.meta.latest_step}. {IS_SEALED ? "Metrics cover the full sealed sets; cards show deterministic diagnostic samples." : "This snapshot refreshes after each completed training-stage probe."}</p></div>
          </div>
          <div className="metricGrid">
            <MetricCard label={`QA · latest (step ${data.meta.latest_step})`} value={percent(data.qa.metrics[data.meta.latest_label].exact_accuracy)} note={`token F1 ${data.qa.metrics[data.meta.latest_label].mean_token_f1.toFixed(3)}`} tone="teal" />
            <MetricCard label={`QA · ${CHECKPOINT_LABELS.baseline}`} value={percent(data.qa.metrics.baseline.exact_accuracy)} note={`token F1 ${data.qa.metrics.baseline.mean_token_f1.toFixed(3)}`} />
            <MetricCard label={`Grounding · latest (step ${data.meta.latest_step})`} value={percent(data.grounding.metrics[data.meta.latest_label].click_accuracy)} note={`median ${data.grounding.metrics[data.meta.latest_label].median_pixel_distance?.toFixed(1)} px`} tone="teal" />
            <MetricCard label={`Grounding · ${CHECKPOINT_LABELS.baseline}`} value={percent(data.grounding.metrics.baseline.click_accuracy)} note={`median ${data.grounding.metrics.baseline.median_pixel_distance?.toFixed(1)} px`} tone="orange" />
            {data.action && <><MetricCard label={`Actions · latest (step ${data.meta.latest_step})`} value={percent(data.action.metrics[data.meta.latest_label].action_name_accuracy)} note={`click-in-bounds ${percent(data.action.metrics[data.meta.latest_label].click_inside_bbox_accuracy ?? 0)}`} tone="teal" /><MetricCard label={`Actions · ${CHECKPOINT_LABELS.baseline}`} value={percent(data.action.metrics.baseline.action_name_accuracy)} note={`click-in-bounds ${percent(data.action.metrics.baseline.click_inside_bbox_accuracy ?? 0)}`} tone="orange" /></>}
          </div>
        </div>
      </header>

      <section className="taskTabs" aria-label="Dataset task">
        <button className={task === "qa" ? "active" : ""} onClick={() => switchTask("qa")}><span>01</span> Screenshot QA <b>{data.qa.meta.examples}</b></button>
        <button className={task === "grounding" ? "active" : ""} onClick={() => switchTask("grounding")}><span>02</span> GUI grounding <b>{data.grounding.meta.examples}</b></button>
        {data.action && <button className={task === "action" ? "active" : ""} onClick={() => switchTask("action")}><span>03</span> Browser actions <b>{data.action.meta.scored_examples}</b></button>}
      </section>

      <section className="explorer" id="examples">
        <aside className="filters">
          <div><p className="filterLabel">Search</p><input aria-label="Search dataset examples" onChange={(event) => { setQuery(event.target.value); setVisible(PAGE_SIZE); }} placeholder={task === "qa" ? "button, price, headline…" : "settings, link, menu…"} type="search" value={query} /></div>
          {task === "qa" && <div><p className="filterLabel">Question type</p><div className="filterStack"><button className={type === "all" ? "active" : ""} onClick={() => setType("all")}><span>All examples</span><b>{data.qa.meta.examples}</b></button>{Object.entries(data.qa.meta.question_types).map(([name, count]) => <button className={type === name ? "active" : ""} key={name} onClick={() => { setType(name); setVisible(PAGE_SIZE); }}><span>{name}</span><b>{count}</b></button>)}</div></div>}
          {task === "action" && data.action && <div><p className="filterLabel">Target action</p><div className="filterStack"><button className={type === "all" ? "active" : ""} onClick={() => setType("all")}><span>All actions</span><b>{data.action.meta.scored_examples}</b></button>{Object.entries(data.action.meta.target_action_counts).map(([name, count]) => <button className={type === name ? "active" : ""} key={name} onClick={() => { setType(name); setVisible(PAGE_SIZE); }}><span>{name}</span><b>{count}</b></button>)}</div></div>}
          <div><p className="filterLabel">{task === "qa" ? "Latest result" : "Latest click"}</p><div className="filterStack"><button className={status === "all" ? "active" : ""} onClick={() => setStatus("all")}><span>Any result</span></button>{task === "qa" ? <><button className={status === "recovered" ? "active" : ""} onClick={() => setStatus("recovered")}><span>Exact answer</span></button><button className={status === "still-wrong" ? "active" : ""} onClick={() => setStatus("still-wrong")}><span>Needs review</span></button></> : <><button className={status === "hit" ? "active" : ""} onClick={() => setStatus("hit")}><span>Inside target</span></button><button className={status === "miss" ? "active" : ""} onClick={() => setStatus("miss")}><span>Missed target</span></button></>}</div></div>
          <div className="datasetNote"><strong>{task === "qa" ? "What one QA record contains" : "How to read the overlay"}</strong><p>{task === "qa" ? "Screenshot, natural-language question, concise reference answer, QA type, form, domain, and URL." : `Green marks the annotated target. Gray is ${CHECKPOINT_LABELS.baseline}; teal is the latest model point at step ${data.meta.latest_step}.`}</p><small>{data.meta.license}</small></div>
        </aside>

        <div className="results">
          <div className="resultsHeader"><div><p className="eyebrow">{task === "qa" ? "QA records" : task === "grounding" ? "Grounding records" : "Action records"}</p><h2>{filtered.length} matching examples</h2></div>{(query || type !== "all" || status !== "all") && <button className="reset" onClick={resetFilters}>Reset filters</button>}</div>
          <div className="cards">{task === "qa" ? qaFiltered.slice(0, visible).map((record, index) => <QaCard eager={index < 4} key={record.id} record={record} />) : task === "grounding" ? groundFiltered.slice(0, visible).map((record, index) => <GroundingCard eager={index < 4} key={record.id} record={record} />) : actionFiltered.slice(0, visible).map((record, index) => <ActionCard eager={index < 4} key={record.id} record={record} />)}</div>
          {visible < filtered.length && <button className="loadMore" onClick={() => setVisible((value) => value + PAGE_SIZE)}>Load {Math.min(PAGE_SIZE, filtered.length - visible)} more</button>}
        </div>
      </section>
      <footer><span>Spider EXP005 · generated from validated, content-addressed prediction receipts</span><span>MolmoWeb · ODC-BY 1.0</span></footer>
    </main>
  );
}
