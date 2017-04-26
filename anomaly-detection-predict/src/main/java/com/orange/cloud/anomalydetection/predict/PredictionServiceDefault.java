package com.orange.cloud.anomalydetection.predict;

import com.orange.cloud.anomalydetection.predict.domain.Observation;
import com.orange.cloud.anomalydetection.predict.domain.Prediction;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
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
    public Prediction predict(Observation observation) throws Exception {
        //normalize inputs since model has been trained with normalized data
        INDArray input = Nd4j.zeros(1, 1);
        input.putScalar(new int[]{0, 0}, observation.getValue());
        normalizer.transform(input);
        INDArray predicted = model.rnnTimeStep(input);
        //Revert data back to original values
        normalizer.revertFeatures(input);
        normalizer.revertLabels(predicted);
        return new Prediction(predicted.getDouble(0));
    }

    @Override
    public INDArray predict(INDArray observation) throws Exception {
        normalizer.transform(observation);
        INDArray prediction = model.rnnTimeStep(observation);
        //Revert data back to original values
        normalizer.revertFeatures(observation);
        normalizer.revertLabels(prediction);
        return prediction;
    }


}
