-- | @futhark metal@
module Futhark.CLI.Metal (main) where

import Futhark.Actions (compileMetalAction)
import Futhark.Compiler.CLI
import Futhark.Passes (gpumemPipeline)

-- | Run @futhark metal@
main :: String -> [String] -> IO ()
main = compilerMain
  ()
  []
  "Compile Metal"
  "Generate C and MSL code calling Metal from optimised Futhark program."
  gpumemPipeline
  $ \fcfg () mode outpath prog ->
    actionProcedure (compileMetalAction fcfg mode outpath) prog
