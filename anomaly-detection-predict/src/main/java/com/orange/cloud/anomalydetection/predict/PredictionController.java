package com.orange.cloud.anomalydetection.predict;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * @author Sebastien Bortolussi
 */
@RestController("/predict")
public class PredictionController {

    PredictionService predictionService;

    public PredictionController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @GetMapping
    String predict(@RequestParam String value) throws Exception {
        INDArray input = Nd4j.zeros(1, 1);
        input.putScalar(new int[]{0, 0}, new Double(value));
        final INDArray predicted = predictionService.predict(input);
        return String.valueOf(predicted.getDouble(0));
    }

}
