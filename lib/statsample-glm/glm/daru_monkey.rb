module Daru
  class DataFrame
    # getting errors without float64
    def to_nmatrix
      each_vector.select do |vector|
        vector.numeric? && !vector.include_values?(*Daru::MISSING_VALUES)
      end.map(&:to_a).transpose.to_nm(nil, :float64)
    end
  end

  class Vector
    # yaml serialization was broked
    def respond_to_missing?(name, include_private = false)
      name.to_s.end_with?('=') || (@index && has_index?(name)) || super
    end
  end
end
