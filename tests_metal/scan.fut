-- Inclusive prefix sum, exercising the two-pass scan.
-- ==
-- input { [1i32, 2i32, 3i32, 4i32, 5i32] }
-- output { [1i32, 3i32, 6i32, 10i32, 15i32] }
-- compiled random input { [1000000]i32 } auto output

def main (xs: []i32) : []i32 = scan (+) 0 xs
