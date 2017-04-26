package com.orange.cloud.anomalydetection.predict;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.springframework.stereotype.Component;

@Component
public class PredictionServiceDefault implements PredictionService {

    private NormalizerMinMaxScaler normalizer;

    private MultiLayerNetwork model;

    public PredictionServiceDefault(NormalizerMinMaxScaler normalizer, MultiLayerNetwork model) {
        this.normalizer = normalizer;
        this.model = model;
    }

    @Override
    public INDArray predict(INDArray featureMatrix) throws Exception {
        //normalize inputs since model has been trained with normalized data
        normalizer.transform(featureMatrix);
        INDArray predicted = model.rnnTimeStep(featureMatrix);
        //Revert data back to original values
        normalizer.revertFeatures(featureMatrix);
        normalizer.revertLabels(predicted);
        return predicted;
    }


}
