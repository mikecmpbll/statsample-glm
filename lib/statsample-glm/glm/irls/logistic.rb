require 'statsample-glm/glm/irls/base'

module Statsample
  module GLM
    module IRLS
      class Logistic < Statsample::GLM::IRLS::Base
        def initialize(*)
          super
        end

        def to_s
          "Statsample::GLM::Logistic"
        end

      protected
        def measurement x, b
          x.dot(b).map{ |y| 1.fdiv(1 + Math.exp(-y)) }
        end

        def weight mus
          weights = mus.map{ |p| p * (1 - p) }

          NMatrix.diagonal(weights)
        end

        def jacobian x, y, mus
          column_data = y.map.with_index{ |y, i| y - mus[i] }

          x.transpose.dot(NMatrix.new([column_data.size, 1], column_data))
        end

        def hessian x, mus
          x.transpose.dot(weight(mus)).dot(x) * -1
        end
      end
    end
  end
end
