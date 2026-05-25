import type { Metrics } from "../state/reviewStore";

type Props = {
  metrics: Record<string, Metrics>;
};

const labels: Record<string, string> = {
  finetuned: "Fine-tuned",
  base: "Base",
  gpt4o: "GPT-4o"
};

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function MetricTable({ metrics }: Props) {
  return (
    <section className="panel">
      <div className="panelHeader">
        <h2>Model Comparison</h2>
      </div>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Quality</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(metrics).map(([key, value]) => (
            <tr key={key}>
              <td>{labels[key] ?? key}</td>
              <td>{formatPercent(value.precision)}</td>
              <td>{formatPercent(value.recall)}</td>
              <td>{formatPercent(value.f1)}</td>
              <td>{value.quality.toFixed(1)}/3</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
