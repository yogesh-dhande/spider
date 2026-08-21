"use client";

import { useMemo, useState } from "react";
import datasetJson from "./exp005-data.json";

type Counts = Record<string, number>;
type Website = {
  domain: string;
  category: string;
  confidence: string;
  examples: number;
  tasks: Counts;
  sources: Counts;
  surfaces: Counts;
};
type TrainingRecord = {
  id: string;
  task: "qa" | "grounding" | "action";
  domain: string;
  surface: string;
  category: string;
  category_confidence: string;
  source: string;
  url: string;
  question: string;
  answer: string;
  question_type: string | null;
  target_action: Record<string, string | number> | null;
  bbox_normalized: [number, number, number, number] | null;
  target_point_normalized: [number, number] | null;
  original_width: number | null;
  original_height: number | null;
  image: string | null;
  image_available: boolean;
  image_source_file: string;
};
type DatasetPayload = {
  meta: {
    provenance: string;
    inventory_identity: string;
    sample_seed: number;
    sample_examples: number;
    image_examples: number;
    license: string;
  };
  summary: {
    examples: number;
    websites: number;
    task_counts: Counts;
    category_counts: Counts;
    category_website_counts: Counts;
  };
  websites: Website[];
  records: TrainingRecord[];
};

const dataset = datasetJson as DatasetPayload;
const CATEGORIES = [
  "work_application",
  "transactional_application",
  "service_application",
  "content_reference",
  "general_web",
];
const TASKS = ["action", "grounding", "qa"];

function label(value: string) {
  return value.replaceAll("_", " ");
}

function compact(value: number) {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function TrainingImage({ record }: { record: TrainingRecord }) {
  if (!record.image) {
    const reason = record.task === "qa"
      ? "QA images remain inside large Arrow shards; prompt and answer are still browsable."
      : "This preview was not fetched for the compact dashboard sample; its source locator is retained.";
    return (
      <div className="trainingPlaceholder">
        <span>Screenshot locator retained</span>
        <strong>Preview not materialized</strong>
        <small>{reason}</small>
      </div>
    );
  }
  const bbox = record.bbox_normalized;
  const grounding = record.task === "grounding";
  const scale = grounding ? 0.1 : 100;
  const target = grounding
    ? record.target_point_normalized
    : record.target_action?.name === "click"
      ? [Number(record.target_action.x), Number(record.target_action.y)] as [number, number]
      : null;
  return (
    <div className="trainingImage">
      <img alt={`Training screenshot from ${record.domain}`} loading="lazy" src={record.image} />
      {bbox && <span className="targetBox" style={{ left: `${bbox[0] * scale}%`, top: `${bbox[1] * scale}%`, width: `${(bbox[2] - bbox[0]) * scale}%`, height: `${(bbox[3] - bbox[1]) * scale}%` }} />}
      {target && Number.isFinite(target[0]) && Number.isFinite(target[1]) && <span className="clickMarker targetPoint" style={{ left: `${target[0] * (grounding ? 0.1 : 1)}%`, top: `${target[1] * (grounding ? 0.1 : 1)}%` }}>GT</span>}
      <div className="imageMeta"><span>{record.domain}</span><span>{record.original_width} × {record.original_height}</span></div>
    </div>
  );
}

function TrainingCard({ record }: { record: TrainingRecord }) {
  return (
    <article className="recordCard trainingCard">
      <TrainingImage record={record} />
      <div className="recordBody">
        <div className="recordTags"><span>{record.task}</span><span>{label(record.category)}</span><span>{record.source}</span>{record.question_type && <span>{record.question_type}</span>}</div>
        <h3>{record.question}</h3>
        <div className="answerBlock reference"><p>{record.task === "action" ? "Target action" : "Training answer"}</p><div>{record.task === "action" ? JSON.stringify(record.target_action) : record.answer}</div></div>
        <div className="sampleMeta"><span>surface</span><b>{record.surface}</b><span>classification</span><b>{record.category_confidence}</b></div>
        {record.url && <a className="sourceLink" href={record.url} rel="noreferrer" target="_blank">Open source URL ↗</a>}
      </div>
    </article>
  );
}

export function TrainingDataView({ onDiagnostics }: { onDiagnostics: () => void }) {
  const [task, setTask] = useState("all");
  const [category, setCategory] = useState("all");
  const [query, setQuery] = useState("");
  const [websiteCategory, setWebsiteCategory] = useState("all");
  const [websiteQuery, setWebsiteQuery] = useState("");
  const [websiteLimit, setWebsiteLimit] = useState(80);

  const records = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return dataset.records.filter((record) =>
      (task === "all" || record.task === task) &&
      (category === "all" || record.category === category) &&
      (!needle || [record.question, record.answer, record.domain, record.surface, record.source].join(" ").toLowerCase().includes(needle))
    );
  }, [category, query, task]);
  const websites = useMemo(() => {
    const needle = websiteQuery.trim().toLowerCase();
    return dataset.websites.filter((website) =>
      (websiteCategory === "all" || website.category === websiteCategory) &&
      (!needle || [website.domain, ...Object.keys(website.surfaces), ...Object.keys(website.sources)].join(" ").toLowerCase().includes(needle))
    );
  }, [websiteCategory, websiteQuery]);
  const maxCategory = Math.max(...Object.values(dataset.summary.category_counts));
  const appExamples = CATEGORIES.slice(0, 3).reduce((total, name) => total + (dataset.summary.category_counts[name] ?? 0), 0);

  return (
    <main>
      <header className="dataHero">
        <nav><div className="brandMark">SP</div><div className="brandText"><strong>Spider Lab</strong><span>EXP005 · data audit</span></div><div className="viewSwitch"><button className="active">Training data</button><button onClick={onDiagnostics}>Model diagnostics</button></div></nav>
        <div className="dataHeroGrid">
          <div><p className="eyebrow">MolmoWeb · {dataset.meta.provenance}</p><h1>Know what<br />we train on.</h1><p className="dek">Browse the website mix, task coverage, and a deterministic cross-category sample before trusting an ablation result.</p><div className="diagnosis"><span className="pulse" /><p><strong>Reproducible sample:</strong> seed {dataset.meta.sample_seed}; {dataset.meta.sample_examples} records, including {dataset.meta.image_examples} materialized visual previews.</p></div></div>
          <div className="metricGrid dataMetrics">
            <div className="metricCard teal"><span>Training candidates</span><strong>{compact(dataset.summary.examples)}</strong><small>{Object.entries(dataset.summary.task_counts).map(([name, count]) => `${name} ${compact(count)}`).join(" · ")}</small></div>
            <div className="metricCard"><span>Websites represented</span><strong>{dataset.summary.websites.toLocaleString()}</strong><small>registrable domains in the candidate pool</small></div>
            <div className="metricCard"><span>Application examples</span><strong>{(appExamples / dataset.summary.examples * 100).toFixed(1)}%</strong><small>work + transaction + service applications</small></div>
            <div className="metricCard orange"><span>Classification audit</span><strong>{dataset.websites.filter((row) => row.confidence === "unknown").length.toLocaleString()}</strong><small>websites use conservative fallback labels</small></div>
          </div>
        </div>
      </header>

      <section className="categorySection">
        <div className="sectionHeading"><div><p className="eyebrow">Composition</p><h2>Website categories in training candidates</h2></div><p>Bars count examples; the right column counts distinct websites. Categories are transparent heuristics and manual overrides, not claims about site identity.</p></div>
        <div className="categoryBars">{CATEGORIES.map((name) => <div className="categoryRow" key={name}><span>{label(name)}</span><div><i style={{ width: `${Math.max(1, (dataset.summary.category_counts[name] ?? 0) / maxCategory * 100)}%` }} /></div><b>{(dataset.summary.category_counts[name] ?? 0).toLocaleString()}</b><small>{dataset.summary.category_website_counts[name] ?? 0} sites</small></div>)}</div>
      </section>

      <section className="dataExplorer" id="training-examples">
        <div className="sectionHeading"><div><p className="eyebrow">Visual sample</p><h2>Browse training examples</h2></div><p>{records.length} of {dataset.records.length} deterministic sample records match the filters.</p></div>
        <div className="inlineFilters">
          <input aria-label="Search training examples" onChange={(event) => setQuery(event.target.value)} placeholder="Search prompt, answer, website…" type="search" value={query} />
          <select aria-label="Filter by task" onChange={(event) => setTask(event.target.value)} value={task}><option value="all">All tasks</option>{TASKS.map((name) => <option key={name} value={name}>{label(name)}</option>)}</select>
          <select aria-label="Filter by website category" onChange={(event) => setCategory(event.target.value)} value={category}><option value="all">All categories</option>{CATEGORIES.map((name) => <option key={name} value={name}>{label(name)}</option>)}</select>
        </div>
        <div className="cards">{records.map((record) => <TrainingCard key={record.id} record={record} />)}</div>
        {!records.length && <div className="emptyState">No sample examples match these filters.</div>}
      </section>

      <section className="websiteSection" id="websites">
        <div className="sectionHeading"><div><p className="eyebrow">Website inventory</p><h2>{websites.length.toLocaleString()} matching websites</h2></div><p>Counts reflect training candidates, before the final nested ladder is selected.</p></div>
        <div className="inlineFilters websiteFilters"><input aria-label="Search websites" onChange={(event) => { setWebsiteQuery(event.target.value); setWebsiteLimit(80); }} placeholder="Search domain, surface, source…" type="search" value={websiteQuery} /><select aria-label="Filter websites by category" onChange={(event) => { setWebsiteCategory(event.target.value); setWebsiteLimit(80); }} value={websiteCategory}><option value="all">All categories</option>{CATEGORIES.map((name) => <option key={name} value={name}>{label(name)}</option>)}</select></div>
        <div className="websiteTableWrap"><table><thead><tr><th>Website</th><th>Category</th><th>Examples</th><th>Task mix</th><th>Top surfaces</th><th>Confidence</th></tr></thead><tbody>{websites.slice(0, websiteLimit).map((website) => <tr key={website.domain}><td><strong>{website.domain}</strong><small>{Object.keys(website.sources).join(" · ")}</small></td><td><span className={`categoryPill ${website.category}`}>{label(website.category)}</span></td><td>{website.examples.toLocaleString()}</td><td>{Object.entries(website.tasks).map(([name, count]) => <span className="countPair" key={name}>{name} <b>{compact(count)}</b></span>)}</td><td>{Object.keys(website.surfaces).slice(0, 3).join(" · ")}</td><td>{website.confidence}</td></tr>)}</tbody></table></div>
        {websiteLimit < websites.length && <button className="loadMore" onClick={() => setWebsiteLimit((value) => value + 80)}>Show 80 more websites</button>}
      </section>
      <footer><span>Spider EXP005 · inventory {dataset.meta.inventory_identity.slice(0, 12)}</span><span>MolmoWeb · {dataset.meta.license}</span></footer>
    </main>
  );
}
