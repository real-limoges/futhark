{-# LANGUAGE QuasiQuotes #-}

-- | Code generation for Metal.
module Futhark.CodeGen.Backends.CMetal
  ( compileProg,
    GC.CParts (..),
    GC.asLibrary,
    GC.asExecutable,
    GC.asServer,
  )
where

import Data.Map qualified as M
import Data.Text qualified as T
import Futhark.CodeGen.Backends.GPU
import Futhark.CodeGen.Backends.GenericC qualified as GC
import Futhark.CodeGen.Backends.GenericC.Options
import Futhark.CodeGen.ImpCode.OpenCL
import Futhark.CodeGen.ImpGen.Metal qualified as ImpGen
import Futhark.CodeGen.RTS.C (backendsMetalH)
import Futhark.IR.GPUMem hiding
  ( CmpSizeLe,
    GetSize,
    GetSizeMax,
  )
import Futhark.MonadFreshNames
import Language.C.Quote.OpenCL qualified as C
import NeatInterpolation (untrimming)

mkBoilerplate ::
  T.Text ->
  [(Name, KernelConstExp)] ->
  M.Map Name KernelSafety ->
  [PrimType] ->
  [FailureMsg] ->
  GC.CompilerM OpenCL () ()
mkBoilerplate metal_program macros kernels types failures = do
  generateGPUBoilerplate
    metal_program
    macros
    backendsMetalH
    (M.keys kernels)
    types
    failures

  GC.headerDecl GC.InitDecl [C.cedecl|void futhark_context_config_set_device(struct futhark_context_config *cfg, const char* s);|]
  GC.headerDecl GC.InitDecl [C.cedecl|const char* futhark_context_config_get_program(struct futhark_context_config *cfg);|]
  GC.headerDecl GC.InitDecl [C.cedecl|void futhark_context_config_set_program(struct futhark_context_config *cfg, const char* s);|]

cliOptions :: [Option]
cliOptions =
  gpuOptions
    ++ [ Option
           { optionLongName = "dump-metal",
             optionShortName = Nothing,
             optionArgument = RequiredArgument "FILE",
             optionDescription = "Dump the embedded Metal (MSL) kernels to the indicated file.",
             optionAction =
               [C.cstm|{const char* prog = futhark_context_config_get_program(cfg);
                        if (dump_file(optarg, prog, strlen(prog)) != 0) {
                          fprintf(stderr, "%s: %s\n", optarg, strerror(errno));
                          exit(1);
                        }
                        exit(0);}|]
           },
         Option
           { optionLongName = "load-metal",
             optionShortName = Nothing,
             optionArgument = RequiredArgument "FILE",
             optionDescription = "Instead of using the embedded Metal (MSL) kernels, load them from the indicated file.",
             optionAction =
               [C.cstm|{ size_t n; const char *s = slurp_file(optarg, &n);
                         if (s == NULL) { fprintf(stderr, "%s: %s\n", optarg, strerror(errno)); exit(1); }
                         futhark_context_config_set_program(cfg, s);
                       }|]
           }
       ]

metalMemoryType :: GC.MemoryType OpenCL ()
metalMemoryType "device" = pure [C.cty|typename gpu_mem|]
metalMemoryType space = error $ "GPU backend does not support '" ++ space ++ "' memory space."

-- | Compile the program to C with calls to Metal (through the futmtl
-- shim; see rts/metal/DESIGN.md).
compileProg :: (MonadFreshNames m) => T.Text -> Prog GPUMem -> m (ImpGen.Warnings, GC.CParts)
compileProg version prog = do
  ( ws,
    Program metal_code metal_prelude macros kernels types failures prog'
    ) <-
    ImpGen.compileProg prog
  (ws,)
    <$> GC.compileProg
      "metal"
      version
      operations
      (mkBoilerplate (metal_prelude <> metal_code) macros kernels types failures)
      metal_includes
      (Space "device", [Space "device", DefaultSpace])
      cliOptions
      prog'
  where
    operations :: GC.Operations OpenCL ()
    operations =
      gpuOperations
        { GC.opsMemoryType = metalMemoryType
        }
    -- Metal is reached through the C-callable futmtl shim declared in
    -- backends/metal.h; no Metal system headers exist for C.
    metal_includes =
      [untrimming|
       #include <stdint.h>
      |]
