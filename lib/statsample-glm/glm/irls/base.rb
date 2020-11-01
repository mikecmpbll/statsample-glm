module Statsample
  module GLM
    module IRLS
      class Base
        attr_accessor :coefficients, :standard_errors, :iterations,
          :fitted_mean_values, :residuals, :degrees_of_freedom

        def initialize data_set, dependent, opts = {}, load_data = nil
          @opts = opts
          @data_set = data_set.to_nmatrix
          @dependent = dependent

          load_data ? load(load_data) : irls
        end

        def data
          {
            coefficients: @coefficients,
            iterations: @iterations,
            standard_errors: @standard_errors,
            fitted_mean_values: @fitted_mean_values,
            residuals: @residuals,
            degrees_of_freedom: @degrees_of_freedom,
          }
        end

        def load(data)
          @coefficients = data[:coefficients]
          @iterations = data[:iterations]
          @standard_errors = data[:standard_errors]
          @fitted_mean_values = data[:fitted_mean_values]
          @residuals = data[:residuals]
          @degrees_of_freedom = data[:degrees_of_freedom]
        end

        # Use the fitted GLM to obtain predictions on new data.
        #
        # == Arguments
        #
        # * new_data - a `Daru::DataFrame` containing new observations for the same
        #   variables that were used to fit the model. The vectors must be given
        #   in the same order as in the data frame that was originally used to fit
        #   the model. If `new_data` is not provided, then the original data frame
        #   which was used to fit the model, is used in place of `new_data`.
        #
        # == Returns
        #
        #   A `Daru::Vector` containing the predictions. The predictions are
        #   computed on the scale of the response variable (for example, for
        #   the logistic regression model, the predictions are probabilities
        #   on logit scale).
        #
        def predict new_data_set=nil
          if new_data_set.nil?
            @fitted_mean_values
          else
            new_data_matrix = new_data_set.to_nmatrix
            b = @coefficients.to_nmatrix :vertical
            Daru::Vector.new measurement(new_data_matrix, b).entries
          end
        end

      private
        def irls
          b = NMatrix.new([@data_set.cols, 1], 0)

          1.upto(@opts[:iterations]) do |i|
            mus = measurement(@data_set, b).transpose.entries
            intermediate = hessian(@data_set, mus).inverse.dot(
              jacobian(@data_set, @dependent, mus)
            )
            b_new = b - intermediate

            delta = (b_new - b).map(&:abs).entries.sum - 1
            if progress?
              puts "iteration #{i}, delta #{delta} (∂ #{delta - @opts[:epsilon]})"
            end

            complete = delta < @opts[:epsilon]
            b = b_new

            break if complete
          end

          @coefficients = Daru::Vector.new(b.column(0).entries)
          @iterations = @opts[:iterations]
          mus = measurement(@data_set, b).transpose.entries
          @standard_errors = Daru::Vector.new(
            hessian(@data_set, mus).inverse
              .diagonal
              .map{ |x| -x }
              .map{ |y| Math.sqrt(y) }
          )
          @fitted_mean_values = Daru::Vector.new measurement(@data_set, b).entries
          @residuals = @dependent - @fitted_mean_values
          @degrees_of_freedom = @dependent.count - @data_set.cols
        end

        def progress?
          ENV['PROGRESS']
        end
      end
    end
  end
end
