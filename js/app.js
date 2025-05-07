let modelo;
let datosPerdida = [];
let grafico;

function crearModelo() {
  modelo = tf.sequential();
  modelo.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  modelo.compile({ loss: "meanSquaredError", optimizer: "sgd" });
}

async function entrenarModelo() {
  crearModelo();

  const xs = tf.tensor1d([1, 2, 3, 4, 5]);
  const ys = tf.tensor1d([3, 6, 9, 12, 15]);

  datosPerdida = [];

  await modelo.fit(xs, ys, {
    epochs: 100,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        datosPerdida.push({ epoch, loss: logs.loss });
      },
    },
  });

  graficarPerdida();

  document.getElementById("resultado").innerHTML =
    "<p><strong>Estado:</strong> Modelo entrenado correctamente</p>";
}

function graficarPerdida() {
  const ctx = document.getElementById("grafico-perdida").getContext("2d");

  const labels = datosPerdida.map((d) => d.epoch);
  const losses = datosPerdida.map((d) => d.loss);

  if (grafico) grafico.destroy();

  grafico = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Pérdida (Loss)",
          data: losses,
          borderColor: "cyan",
          fill: false,
          tension: 0.1,
          pointRadius: 3,
          pointHoverRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: {
          title: {
            display: true,
            text: "Época",
          },
        },
        y: {
          title: {
            display: true,
            text: "Valor de Pérdida",
          },
        },
      },
    },
  });

  const perdidaInicial = losses[0].toFixed(4);
  const perdidaFinal = losses[losses.length - 1].toFixed(4);
  const reduccion = (
    ((losses[0] - losses[losses.length - 1]) / losses[0]) *
    100
  ).toFixed(2);

  document.getElementById(
    "info-perdida"
  ).innerText = `Pérdida inicial: ${perdidaInicial}, Pérdida final: ${perdidaFinal} (Reducción: ${reduccion}%)`;
}

async function predecir() {
  const input = document.getElementById("input-valores").value;
  const valoresX = input.split(",").map((x) => parseFloat(x.trim()));
  const tensorX = tf.tensor2d(valoresX, [valoresX.length, 1]);

  const predicciones = modelo.predict(tensorX);
  const valoresY = await predicciones.data();

  let html = "<h4>Resultados:</h4><ul>";
  for (let i = 0; i < valoresX.length; i++) {
    html += `<li>Para x = ${valoresX[i]}: y = ${valoresY[i].toFixed(2)}</li>`;
  }
  html += "</ul>";
  document.getElementById("resultado").innerHTML += html;
}
