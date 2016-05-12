describe Statsample::GLM::Normal do
  context "MLE algorithm" do
    before do
      # Below data set taken from http://dl.dropbox.com/u/10246536/Web/RTutorialSeries/dataset_multipleRegression.csv
      @ds = Daru::DataFrame.from_csv "spec/data/normal.csv", 
        order: ['ROLL', 'UNEM', 'HGRAD', 'INC']
    end

    it "reports correct values as a Daru::Vector", focus: true do
      @glm = Statsample::GLM.compute @ds, 'ROLL', :normal, {algorithm: :mle}
      
      expect_similar_vector @glm.coefficients, [450.12450365911894, 
        0.4064837278023981, 4.27485769721736, -9153.254462671905]
    end

    it "reports correct values when constant is different from 1", focus: true do
      @glm = Statsample::GLM.compute @ds, 'ROLL', :normal, {constant: 2, algorithm: :mle}
      
      expect_similar_vector @glm.coefficients, [450.12450365911894, 
        0.4064837278023981, 4.27485769721736, -4576.627231335952]
    end

    it "computes predictions of new data correctly" do
      @glm = Statsample::GLM.compute @ds, 'ROLL', :normal, {algorithm: :mle}
      new_data = Daru::DataFrame.new([[7, 8, 9],
                                      [15000, 16000, 17000],
                                      [3000, 4000, 5000]],
                                     order: ['UNEM', 'HGRAD', 'INC'])
      # predictions obtained with predict.lm in R:
      predictions = [12919.44607162998, 18050.91200030885,
                     23182.37792898773]
      expect_similar_vector @glm.predict(new_data), predictions
    end
  end
end
