async function treinarEPrever() {

    // Pegando elementos da tela

    const textoStatus = document.getElementById("status");

    const textoResultado = document.getElementById("resultado");

    // Pegando valor digitado pelo usuário

    const anosExperiencia = Number(document.getElementById("experiencia").value);

    textoStatus.innerText = "Status: Treinando a IA...";

    // =========================

    // 1. CRIAR O MODELO

    // =========================

    const modelo = tf.sequential();

    modelo.add(tf.layers.dense({

        units: 1,

        inputShape: [1]

    }));

    // =========================

    // 2. COMPILAR O MODELO

    // =========================

    modelo.compile({

        loss: 'meanSquaredError',

        optimizer: 'sgd'

    });

    // =========================

    // 3. DADOS DE TREINO

    // X = anos de experiência

    // Y = salário

    // =========================

    const dadosEntrada = tf.tensor2d([1, 2, 3, 4, 5], [5, 1]);

    const dadosSaida = tf.tensor2d([1500, 2500, 3500, 4500, 5500], [5, 1]);

    // =========================

    // 4. TREINAMENTO

    // =========================

    await modelo.fit(dadosEntrada, dadosSaida, {

        epochs: 200

    });

    textoStatus.innerText = "Status: IA treinada!";

    // =========================

    // 5. PREVISÃO

    // =========================

    const previsao = modelo.predict(

        tf.tensor2d([anosExperiencia], [1, 1])

    );

    // Convertendo resultado para número

    const valorPrevisto = previsao.dataSync()[0];

    // Mostrando resultado na tela

    textoResultado.innerText =

        "Salário Previsto: R$ " + valorPrevisto.toFixed(2);

}
 