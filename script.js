let dataset = [];
let modelo;
const etiquetas = ["manzana", "banana", "pera", "naranja", "uva"]; // Reemplázalo con tus clases reales

/** Carga imágenes y etiquetas desde el input */
async function leerArchivos(files) {
  return new Promise((resolve) => {
    let procesadas = 0;

    for (const file of files) {
      const lector = new FileReader();
      lector.onload = function (e) {
        const img = new Image();
        img.onload = function () {
          const label = file.webkitRelativePath.split("/")[1];
          dataset.push({ img, label });

          procesadas++;
          if (procesadas === files.length) resolve();
        };
        img.src = e.target.result;
      };
      lector.readAsDataURL(file);
    }
  });
}

/** Entrena el modelo con las imágenes cargadas */
async function entrenarModelo() {
  const files = document.getElementById("upload").files;
  if (files.length === 0) {
    alert("❌ No se han seleccionado imágenes para entrenar.");
    return;
  }

  await leerArchivos(files);

  const etiquetasUnicas = [...new Set(dataset.map((d) => d.label))];
  const etiquetaToIndex = Object.fromEntries(etiquetasUnicas.map((e, i) => [e, i]));

  const xs = [];
  const ys = [];

  dataset.forEach(({ img, label }) => {
    const tensorImg = tf.browser.fromPixels(img).resizeNearestNeighbor([64, 64]).toFloat().div(255.0);
    xs.push(tensorImg);
    ys.push(etiquetaToIndex[label]);
  });

  const xsStacked = tf.stack(xs);
  const ysOneHot = tf.oneHot(tf.tensor1d(ys, "int32"), etiquetasUnicas.length);

  modelo = tf.sequential();
  modelo.add(tf.layers.conv2d({ inputShape: [64, 64, 3], filters: 16, kernelSize: 3, activation: "relu" }));
  modelo.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  modelo.add(tf.layers.flatten());
  modelo.add(tf.layers.dense({ units: 64, activation: "relu" }));
  modelo.add(tf.layers.dense({ units: etiquetasUnicas.length, activation: "softmax" }));

  modelo.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });

  await modelo.fit(xsStacked, ysOneHot, {
    epochs: 20,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        document.getElementById("output").textContent += `Época ${epoch + 1}: Precisión = ${logs.acc.toFixed(4)}\n`;
      },
    },
  });

  await modelo.save("indexeddb://modelo-frutas");
  alert("✅ Modelo entrenado y guardado exitosamente.");
}

/** Carga el modelo desde IndexedDB para la clasificación */
async function cargarModelo() {
  try {
    modelo = await tf.loadLayersModel("indexeddb://modelo-frutas");
    document.getElementById("resultado").textContent = "✅ Modelo cargado correctamente.";
  } catch (e) {
    document.getElementById("resultado").textContent = "❌ No se pudo cargar el modelo. Asegúrate de haberlo entrenado antes.";
  }
}

cargarModelo();

/** Clasifica una imagen cargada */
document.getElementById("imagen").addEventListener("change", async function (e) {
  const archivo = e.target.files[0];
  if (!archivo) return;

  const img = new Image();
  img.src = URL.createObjectURL(archivo);
  document.getElementById("preview").src = img.src;

  img.onload = async () => {
    const tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([64, 64]).toFloat().div(255.0).expandDims();

    const pred = modelo.predict(tensor);
    const index = (await pred.argMax(1).data())[0];
    const prob = (await pred.max().data())[0];
    console.log(etiquetas);
    document.getElementById("resultado").textContent = `🔍 Detectado: ${etiquetas[index]} (${(prob * 100).toFixed(2)}% seguro)`;
  };
});
