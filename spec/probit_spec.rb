describe Statsample::GLM::Probit do
  context "IRLS algorithm" do
    # TODO : Implement this!
  end

  context "MLE algorithm" do
    before do
      @data_set = Daru::DataFrame.from_csv 'spec/data/logistic_mle.csv'
      @data_set.vectors = Daru::Index.new([:a,:b,:c,:y])
      @glm      = Statsample::GLM.compute @data_set, :y, :probit, 
                    {algorithm: :mle, constant: 1}
    end

    it "reports correct values as an array" do
      expect_similar_vector(@glm.coefficients,[0.1763,0.4483,-0.2240,-3.0670],0.001)

      expect(@glm.log_likelihood).to be_within(0.0001).of(-38.31559)
    end

    it "computes predictions on new data correctly" do
      new_data = Daru::DataFrame.new([[-50.0, -100.0], [50.0, 100.0], [50.0, 100.0]],
                                     order: ['a', 'b', 'c'])
      #predictions obtained with in R predict.glm with type='response':
      predictions = [0.2516918644447207, 0.9580621633922622]
      expect_similar_vector @glm.predict(new_data), predictions, delta=1e-4
    end
  end
end
