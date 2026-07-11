-- Reduction over i32, exercising two-stage reduction with
-- threadgroup memory and barriers.
-- ==
-- input { [1i32, 2i32, 3i32, 4i32, 5i32, 6i32, 7i32, 8i32, 9i32, 10i32] }
-- output { 55i32 }
-- compiled random input { [1000000]i32 } auto output

def main (xs: []i32) : i32 = reduce (+) 0 xs
