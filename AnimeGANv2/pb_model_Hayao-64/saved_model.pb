
â
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Ý
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%ÍĖL>"
Ttype0:
2

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

	MirrorPad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	"&
modestring:
REFLECT	SYMMETRIC
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape

ResizeBilinear
images"T
size
resized_images"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
2
StopGradient

input"T
output"T"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.15.02v1.15.0-rc3-22-g590d6ee·ŧ

generator_inputPlaceholder*-
shape$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
dtype0*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ

&generator/G_MODEL/A/MirrorPad/paddingsConst*
_output_shapes

:*9
value0B."                             *
dtype0
Ę
generator/G_MODEL/A/MirrorPad	MirrorPadgenerator_input&generator/G_MODEL/A/MirrorPad/paddings*
	Tpaddings0*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0*
mode	REFLECT
Ņ
Cgenerator/G_MODEL/A/Conv/weights/Initializer/truncated_normal/shapeConst*%
valueB"             *3
_class)
'%loc:@generator/G_MODEL/A/Conv/weights*
dtype0*
_output_shapes
:
ž
Bgenerator/G_MODEL/A/Conv/weights/Initializer/truncated_normal/meanConst*3
_class)
'%loc:@generator/G_MODEL/A/Conv/weights*
_output_shapes
: *
valueB
 *    *
dtype0
ū
Dgenerator/G_MODEL/A/Conv/weights/Initializer/truncated_normal/stddevConst*3
_class)
'%loc:@generator/G_MODEL/A/Conv/weights*
dtype0*
valueB
 *A/>*
_output_shapes
: 
ą
Mgenerator/G_MODEL/A/Conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalCgenerator/G_MODEL/A/Conv/weights/Initializer/truncated_normal/shape*
seed2 *3
_class)
'%loc:@generator/G_MODEL/A/Conv/weights*&
_output_shapes
: *

seed *
T0*
dtype0
Ã
Agenerator/G_MODEL/A/Conv/weights/Initializer/truncated_normal/mulMulMgenerator/G_MODEL/A/Conv/weights/Initializer/truncated_normal/TruncatedNormalDgenerator/G_MODEL/A/Conv/weights/Initializer/truncated_normal/stddev*
T0*3
_class)
'%loc:@generator/G_MODEL/A/Conv/weights*&
_output_shapes
: 
ą
=generator/G_MODEL/A/Conv/weights/Initializer/truncated_normalAddAgenerator/G_MODEL/A/Conv/weights/Initializer/truncated_normal/mulBgenerator/G_MODEL/A/Conv/weights/Initializer/truncated_normal/mean*3
_class)
'%loc:@generator/G_MODEL/A/Conv/weights*&
_output_shapes
: *
T0
Ų
 generator/G_MODEL/A/Conv/weights
VariableV2*
dtype0*3
_class)
'%loc:@generator/G_MODEL/A/Conv/weights*
shared_name *
	container *
shape: *&
_output_shapes
: 
Ą
'generator/G_MODEL/A/Conv/weights/AssignAssign generator/G_MODEL/A/Conv/weights=generator/G_MODEL/A/Conv/weights/Initializer/truncated_normal*
use_locking(*3
_class)
'%loc:@generator/G_MODEL/A/Conv/weights*
validate_shape(*
T0*&
_output_shapes
: 
đ
%generator/G_MODEL/A/Conv/weights/readIdentity generator/G_MODEL/A/Conv/weights*3
_class)
'%loc:@generator/G_MODEL/A/Conv/weights*&
_output_shapes
: *
T0
w
&generator/G_MODEL/A/Conv/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
š
generator/G_MODEL/A/Conv/Conv2DConv2Dgenerator/G_MODEL/A/MirrorPad%generator/G_MODEL/A/Conv/weights/read*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ *
paddingVALID*
data_formatNHWC*
T0*
explicit_paddings
 *
	dilations
*
strides
*
use_cudnn_on_gpu(
ļ
4generator/G_MODEL/A/LayerNorm/beta/Initializer/zerosConst*5
_class+
)'loc:@generator/G_MODEL/A/LayerNorm/beta*
_output_shapes
: *
dtype0*
valueB *    
Å
"generator/G_MODEL/A/LayerNorm/beta
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: *5
_class+
)'loc:@generator/G_MODEL/A/LayerNorm/beta

)generator/G_MODEL/A/LayerNorm/beta/AssignAssign"generator/G_MODEL/A/LayerNorm/beta4generator/G_MODEL/A/LayerNorm/beta/Initializer/zeros*
T0*
_output_shapes
: *5
_class+
)'loc:@generator/G_MODEL/A/LayerNorm/beta*
use_locking(*
validate_shape(
ģ
'generator/G_MODEL/A/LayerNorm/beta/readIdentity"generator/G_MODEL/A/LayerNorm/beta*5
_class+
)'loc:@generator/G_MODEL/A/LayerNorm/beta*
T0*
_output_shapes
: 
đ
4generator/G_MODEL/A/LayerNorm/gamma/Initializer/onesConst*
valueB *  ?*
_output_shapes
: *6
_class,
*(loc:@generator/G_MODEL/A/LayerNorm/gamma*
dtype0
Į
#generator/G_MODEL/A/LayerNorm/gamma
VariableV2*
_output_shapes
: *
shared_name *
	container *
dtype0*
shape: *6
_class,
*(loc:@generator/G_MODEL/A/LayerNorm/gamma

*generator/G_MODEL/A/LayerNorm/gamma/AssignAssign#generator/G_MODEL/A/LayerNorm/gamma4generator/G_MODEL/A/LayerNorm/gamma/Initializer/ones*
T0*6
_class,
*(loc:@generator/G_MODEL/A/LayerNorm/gamma*
use_locking(*
_output_shapes
: *
validate_shape(
ķ
(generator/G_MODEL/A/LayerNorm/gamma/readIdentity#generator/G_MODEL/A/LayerNorm/gamma*
T0*6
_class,
*(loc:@generator/G_MODEL/A/LayerNorm/gamma*
_output_shapes
: 

<generator/G_MODEL/A/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0
ß
*generator/G_MODEL/A/LayerNorm/moments/meanMeangenerator/G_MODEL/A/Conv/Conv2D<generator/G_MODEL/A/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:

2generator/G_MODEL/A/LayerNorm/moments/StopGradientStopGradient*generator/G_MODEL/A/LayerNorm/moments/mean*
T0*&
_output_shapes
:
ä
7generator/G_MODEL/A/LayerNorm/moments/SquaredDifferenceSquaredDifferencegenerator/G_MODEL/A/Conv/Conv2D2generator/G_MODEL/A/LayerNorm/moments/StopGradient*
T0*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ 

@generator/G_MODEL/A/LayerNorm/moments/variance/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0
ĸ
.generator/G_MODEL/A/LayerNorm/moments/varianceMean7generator/G_MODEL/A/LayerNorm/moments/SquaredDifference@generator/G_MODEL/A/LayerNorm/moments/variance/reduction_indices*

Tidx0*&
_output_shapes
:*
T0*
	keep_dims(
r
-generator/G_MODEL/A/LayerNorm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ėž+
Ä
+generator/G_MODEL/A/LayerNorm/batchnorm/addAddV2.generator/G_MODEL/A/LayerNorm/moments/variance-generator/G_MODEL/A/LayerNorm/batchnorm/add/y*&
_output_shapes
:*
T0

-generator/G_MODEL/A/LayerNorm/batchnorm/RsqrtRsqrt+generator/G_MODEL/A/LayerNorm/batchnorm/add*
T0*&
_output_shapes
:
ž
+generator/G_MODEL/A/LayerNorm/batchnorm/mulMul-generator/G_MODEL/A/LayerNorm/batchnorm/Rsqrt(generator/G_MODEL/A/LayerNorm/gamma/read*
T0*&
_output_shapes
: 
Å
-generator/G_MODEL/A/LayerNorm/batchnorm/mul_1Mulgenerator/G_MODEL/A/Conv/Conv2D+generator/G_MODEL/A/LayerNorm/batchnorm/mul*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ *
T0
ū
-generator/G_MODEL/A/LayerNorm/batchnorm/mul_2Mul*generator/G_MODEL/A/LayerNorm/moments/mean+generator/G_MODEL/A/LayerNorm/batchnorm/mul*
T0*&
_output_shapes
: 
ŧ
+generator/G_MODEL/A/LayerNorm/batchnorm/subSub'generator/G_MODEL/A/LayerNorm/beta/read-generator/G_MODEL/A/LayerNorm/batchnorm/mul_2*
T0*&
_output_shapes
: 
Õ
-generator/G_MODEL/A/LayerNorm/batchnorm/add_1AddV2-generator/G_MODEL/A/LayerNorm/batchnorm/mul_1+generator/G_MODEL/A/LayerNorm/batchnorm/sub*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ *
T0
Ž
generator/G_MODEL/A/LeakyRelu	LeakyRelu-generator/G_MODEL/A/LayerNorm/batchnorm/add_1*
alpha%ÍĖL>*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ *
T0

(generator/G_MODEL/A/MirrorPad_1/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               
Ü
generator/G_MODEL/A/MirrorPad_1	MirrorPadgenerator/G_MODEL/A/LeakyRelu(generator/G_MODEL/A/MirrorPad_1/paddings*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ *
T0*
mode	REFLECT*
	Tpaddings0
Õ
Egenerator/G_MODEL/A/Conv_1/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"          @   *5
_class+
)'loc:@generator/G_MODEL/A/Conv_1/weights
Ā
Dgenerator/G_MODEL/A/Conv_1/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *5
_class+
)'loc:@generator/G_MODEL/A/Conv_1/weights*
_output_shapes
: *
dtype0
Â
Fgenerator/G_MODEL/A/Conv_1/weights/Initializer/truncated_normal/stddevConst*5
_class+
)'loc:@generator/G_MODEL/A/Conv_1/weights*
valueB
 *Â=*
dtype0*
_output_shapes
: 
·
Ogenerator/G_MODEL/A/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalEgenerator/G_MODEL/A/Conv_1/weights/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
: @*
T0*

seed *
seed2 *5
_class+
)'loc:@generator/G_MODEL/A/Conv_1/weights
Ë
Cgenerator/G_MODEL/A/Conv_1/weights/Initializer/truncated_normal/mulMulOgenerator/G_MODEL/A/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalFgenerator/G_MODEL/A/Conv_1/weights/Initializer/truncated_normal/stddev*5
_class+
)'loc:@generator/G_MODEL/A/Conv_1/weights*
T0*&
_output_shapes
: @
đ
?generator/G_MODEL/A/Conv_1/weights/Initializer/truncated_normalAddCgenerator/G_MODEL/A/Conv_1/weights/Initializer/truncated_normal/mulDgenerator/G_MODEL/A/Conv_1/weights/Initializer/truncated_normal/mean*
T0*5
_class+
)'loc:@generator/G_MODEL/A/Conv_1/weights*&
_output_shapes
: @
Ý
"generator/G_MODEL/A/Conv_1/weights
VariableV2*5
_class+
)'loc:@generator/G_MODEL/A/Conv_1/weights*&
_output_shapes
: @*
shape: @*
dtype0*
shared_name *
	container 
Đ
)generator/G_MODEL/A/Conv_1/weights/AssignAssign"generator/G_MODEL/A/Conv_1/weights?generator/G_MODEL/A/Conv_1/weights/Initializer/truncated_normal*
validate_shape(*
T0*&
_output_shapes
: @*5
_class+
)'loc:@generator/G_MODEL/A/Conv_1/weights*
use_locking(
ŋ
'generator/G_MODEL/A/Conv_1/weights/readIdentity"generator/G_MODEL/A/Conv_1/weights*5
_class+
)'loc:@generator/G_MODEL/A/Conv_1/weights*&
_output_shapes
: @*
T0
y
(generator/G_MODEL/A/Conv_1/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ā
!generator/G_MODEL/A/Conv_1/Conv2DConv2Dgenerator/G_MODEL/A/MirrorPad_1'generator/G_MODEL/A/Conv_1/weights/read*
strides
*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
use_cudnn_on_gpu(*
paddingVALID*
T0*
explicit_paddings
 *
data_formatNHWC*
	dilations

ž
6generator/G_MODEL/A/LayerNorm_1/beta/Initializer/zerosConst*7
_class-
+)loc:@generator/G_MODEL/A/LayerNorm_1/beta*
dtype0*
valueB@*    *
_output_shapes
:@
É
$generator/G_MODEL/A/LayerNorm_1/beta
VariableV2*7
_class-
+)loc:@generator/G_MODEL/A/LayerNorm_1/beta*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
	container 

+generator/G_MODEL/A/LayerNorm_1/beta/AssignAssign$generator/G_MODEL/A/LayerNorm_1/beta6generator/G_MODEL/A/LayerNorm_1/beta/Initializer/zeros*
validate_shape(*7
_class-
+)loc:@generator/G_MODEL/A/LayerNorm_1/beta*
_output_shapes
:@*
T0*
use_locking(
đ
)generator/G_MODEL/A/LayerNorm_1/beta/readIdentity$generator/G_MODEL/A/LayerNorm_1/beta*
_output_shapes
:@*7
_class-
+)loc:@generator/G_MODEL/A/LayerNorm_1/beta*
T0
―
6generator/G_MODEL/A/LayerNorm_1/gamma/Initializer/onesConst*
dtype0*8
_class.
,*loc:@generator/G_MODEL/A/LayerNorm_1/gamma*
_output_shapes
:@*
valueB@*  ?
Ë
%generator/G_MODEL/A/LayerNorm_1/gamma
VariableV2*
	container *
shared_name *
_output_shapes
:@*8
_class.
,*loc:@generator/G_MODEL/A/LayerNorm_1/gamma*
shape:@*
dtype0

,generator/G_MODEL/A/LayerNorm_1/gamma/AssignAssign%generator/G_MODEL/A/LayerNorm_1/gamma6generator/G_MODEL/A/LayerNorm_1/gamma/Initializer/ones*
use_locking(*
_output_shapes
:@*8
_class.
,*loc:@generator/G_MODEL/A/LayerNorm_1/gamma*
validate_shape(*
T0
ž
*generator/G_MODEL/A/LayerNorm_1/gamma/readIdentity%generator/G_MODEL/A/LayerNorm_1/gamma*
_output_shapes
:@*
T0*8
_class.
,*loc:@generator/G_MODEL/A/LayerNorm_1/gamma

>generator/G_MODEL/A/LayerNorm_1/moments/mean/reduction_indicesConst*
dtype0*!
valueB"         *
_output_shapes
:
å
,generator/G_MODEL/A/LayerNorm_1/moments/meanMean!generator/G_MODEL/A/Conv_1/Conv2D>generator/G_MODEL/A/LayerNorm_1/moments/mean/reduction_indices*
T0*

Tidx0*&
_output_shapes
:*
	keep_dims(
Ģ
4generator/G_MODEL/A/LayerNorm_1/moments/StopGradientStopGradient,generator/G_MODEL/A/LayerNorm_1/moments/mean*&
_output_shapes
:*
T0
ę
9generator/G_MODEL/A/LayerNorm_1/moments/SquaredDifferenceSquaredDifference!generator/G_MODEL/A/Conv_1/Conv2D4generator/G_MODEL/A/LayerNorm_1/moments/StopGradient*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
T0

Bgenerator/G_MODEL/A/LayerNorm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0

0generator/G_MODEL/A/LayerNorm_1/moments/varianceMean9generator/G_MODEL/A/LayerNorm_1/moments/SquaredDifferenceBgenerator/G_MODEL/A/LayerNorm_1/moments/variance/reduction_indices*
	keep_dims(*&
_output_shapes
:*
T0*

Tidx0
t
/generator/G_MODEL/A/LayerNorm_1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ėž+
Ę
-generator/G_MODEL/A/LayerNorm_1/batchnorm/addAddV20generator/G_MODEL/A/LayerNorm_1/moments/variance/generator/G_MODEL/A/LayerNorm_1/batchnorm/add/y*&
_output_shapes
:*
T0

/generator/G_MODEL/A/LayerNorm_1/batchnorm/RsqrtRsqrt-generator/G_MODEL/A/LayerNorm_1/batchnorm/add*&
_output_shapes
:*
T0
Â
-generator/G_MODEL/A/LayerNorm_1/batchnorm/mulMul/generator/G_MODEL/A/LayerNorm_1/batchnorm/Rsqrt*generator/G_MODEL/A/LayerNorm_1/gamma/read*
T0*&
_output_shapes
:@
Ë
/generator/G_MODEL/A/LayerNorm_1/batchnorm/mul_1Mul!generator/G_MODEL/A/Conv_1/Conv2D-generator/G_MODEL/A/LayerNorm_1/batchnorm/mul*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
T0
Ä
/generator/G_MODEL/A/LayerNorm_1/batchnorm/mul_2Mul,generator/G_MODEL/A/LayerNorm_1/moments/mean-generator/G_MODEL/A/LayerNorm_1/batchnorm/mul*
T0*&
_output_shapes
:@
Á
-generator/G_MODEL/A/LayerNorm_1/batchnorm/subSub)generator/G_MODEL/A/LayerNorm_1/beta/read/generator/G_MODEL/A/LayerNorm_1/batchnorm/mul_2*
T0*&
_output_shapes
:@
Û
/generator/G_MODEL/A/LayerNorm_1/batchnorm/add_1AddV2/generator/G_MODEL/A/LayerNorm_1/batchnorm/mul_1-generator/G_MODEL/A/LayerNorm_1/batchnorm/sub*
T0*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@
°
generator/G_MODEL/A/LeakyRelu_1	LeakyRelu/generator/G_MODEL/A/LayerNorm_1/batchnorm/add_1*
alpha%ÍĖL>*
T0*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@

(generator/G_MODEL/A/MirrorPad_2/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
Þ
generator/G_MODEL/A/MirrorPad_2	MirrorPadgenerator/G_MODEL/A/LeakyRelu_1(generator/G_MODEL/A/MirrorPad_2/paddings*
mode	REFLECT*
	Tpaddings0*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
T0
Õ
Egenerator/G_MODEL/A/Conv_2/weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"      @   @   *5
_class+
)'loc:@generator/G_MODEL/A/Conv_2/weights*
dtype0
Ā
Dgenerator/G_MODEL/A/Conv_2/weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *5
_class+
)'loc:@generator/G_MODEL/A/Conv_2/weights*
_output_shapes
: 
Â
Fgenerator/G_MODEL/A/Conv_2/weights/Initializer/truncated_normal/stddevConst*
dtype0*5
_class+
)'loc:@generator/G_MODEL/A/Conv_2/weights*
_output_shapes
: *
valueB
 *=
·
Ogenerator/G_MODEL/A/Conv_2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalEgenerator/G_MODEL/A/Conv_2/weights/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*5
_class+
)'loc:@generator/G_MODEL/A/Conv_2/weights*&
_output_shapes
:@@*
T0
Ë
Cgenerator/G_MODEL/A/Conv_2/weights/Initializer/truncated_normal/mulMulOgenerator/G_MODEL/A/Conv_2/weights/Initializer/truncated_normal/TruncatedNormalFgenerator/G_MODEL/A/Conv_2/weights/Initializer/truncated_normal/stddev*
T0*5
_class+
)'loc:@generator/G_MODEL/A/Conv_2/weights*&
_output_shapes
:@@
đ
?generator/G_MODEL/A/Conv_2/weights/Initializer/truncated_normalAddCgenerator/G_MODEL/A/Conv_2/weights/Initializer/truncated_normal/mulDgenerator/G_MODEL/A/Conv_2/weights/Initializer/truncated_normal/mean*&
_output_shapes
:@@*
T0*5
_class+
)'loc:@generator/G_MODEL/A/Conv_2/weights
Ý
"generator/G_MODEL/A/Conv_2/weights
VariableV2*&
_output_shapes
:@@*
	container *5
_class+
)'loc:@generator/G_MODEL/A/Conv_2/weights*
shape:@@*
dtype0*
shared_name 
Đ
)generator/G_MODEL/A/Conv_2/weights/AssignAssign"generator/G_MODEL/A/Conv_2/weights?generator/G_MODEL/A/Conv_2/weights/Initializer/truncated_normal*&
_output_shapes
:@@*
validate_shape(*
T0*5
_class+
)'loc:@generator/G_MODEL/A/Conv_2/weights*
use_locking(
ŋ
'generator/G_MODEL/A/Conv_2/weights/readIdentity"generator/G_MODEL/A/Conv_2/weights*5
_class+
)'loc:@generator/G_MODEL/A/Conv_2/weights*&
_output_shapes
:@@*
T0
y
(generator/G_MODEL/A/Conv_2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ā
!generator/G_MODEL/A/Conv_2/Conv2DConv2Dgenerator/G_MODEL/A/MirrorPad_2'generator/G_MODEL/A/Conv_2/weights/read*
explicit_paddings
 *
data_formatNHWC*
T0*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
use_cudnn_on_gpu(*
strides
*
paddingVALID*
	dilations

ž
6generator/G_MODEL/A/LayerNorm_2/beta/Initializer/zerosConst*
_output_shapes
:@*7
_class-
+)loc:@generator/G_MODEL/A/LayerNorm_2/beta*
valueB@*    *
dtype0
É
$generator/G_MODEL/A/LayerNorm_2/beta
VariableV2*
dtype0*
shape:@*7
_class-
+)loc:@generator/G_MODEL/A/LayerNorm_2/beta*
	container *
shared_name *
_output_shapes
:@

+generator/G_MODEL/A/LayerNorm_2/beta/AssignAssign$generator/G_MODEL/A/LayerNorm_2/beta6generator/G_MODEL/A/LayerNorm_2/beta/Initializer/zeros*
use_locking(*7
_class-
+)loc:@generator/G_MODEL/A/LayerNorm_2/beta*
_output_shapes
:@*
T0*
validate_shape(
đ
)generator/G_MODEL/A/LayerNorm_2/beta/readIdentity$generator/G_MODEL/A/LayerNorm_2/beta*7
_class-
+)loc:@generator/G_MODEL/A/LayerNorm_2/beta*
T0*
_output_shapes
:@
―
6generator/G_MODEL/A/LayerNorm_2/gamma/Initializer/onesConst*
_output_shapes
:@*8
_class.
,*loc:@generator/G_MODEL/A/LayerNorm_2/gamma*
dtype0*
valueB@*  ?
Ë
%generator/G_MODEL/A/LayerNorm_2/gamma
VariableV2*8
_class.
,*loc:@generator/G_MODEL/A/LayerNorm_2/gamma*
	container *
_output_shapes
:@*
dtype0*
shared_name *
shape:@

,generator/G_MODEL/A/LayerNorm_2/gamma/AssignAssign%generator/G_MODEL/A/LayerNorm_2/gamma6generator/G_MODEL/A/LayerNorm_2/gamma/Initializer/ones*
_output_shapes
:@*
use_locking(*8
_class.
,*loc:@generator/G_MODEL/A/LayerNorm_2/gamma*
T0*
validate_shape(
ž
*generator/G_MODEL/A/LayerNorm_2/gamma/readIdentity%generator/G_MODEL/A/LayerNorm_2/gamma*
T0*
_output_shapes
:@*8
_class.
,*loc:@generator/G_MODEL/A/LayerNorm_2/gamma

>generator/G_MODEL/A/LayerNorm_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
å
,generator/G_MODEL/A/LayerNorm_2/moments/meanMean!generator/G_MODEL/A/Conv_2/Conv2D>generator/G_MODEL/A/LayerNorm_2/moments/mean/reduction_indices*
	keep_dims(*&
_output_shapes
:*
T0*

Tidx0
Ģ
4generator/G_MODEL/A/LayerNorm_2/moments/StopGradientStopGradient,generator/G_MODEL/A/LayerNorm_2/moments/mean*&
_output_shapes
:*
T0
ę
9generator/G_MODEL/A/LayerNorm_2/moments/SquaredDifferenceSquaredDifference!generator/G_MODEL/A/Conv_2/Conv2D4generator/G_MODEL/A/LayerNorm_2/moments/StopGradient*
T0*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@

Bgenerator/G_MODEL/A/LayerNorm_2/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0

0generator/G_MODEL/A/LayerNorm_2/moments/varianceMean9generator/G_MODEL/A/LayerNorm_2/moments/SquaredDifferenceBgenerator/G_MODEL/A/LayerNorm_2/moments/variance/reduction_indices*
T0*&
_output_shapes
:*

Tidx0*
	keep_dims(
t
/generator/G_MODEL/A/LayerNorm_2/batchnorm/add/yConst*
valueB
 *Ėž+*
_output_shapes
: *
dtype0
Ę
-generator/G_MODEL/A/LayerNorm_2/batchnorm/addAddV20generator/G_MODEL/A/LayerNorm_2/moments/variance/generator/G_MODEL/A/LayerNorm_2/batchnorm/add/y*&
_output_shapes
:*
T0

/generator/G_MODEL/A/LayerNorm_2/batchnorm/RsqrtRsqrt-generator/G_MODEL/A/LayerNorm_2/batchnorm/add*
T0*&
_output_shapes
:
Â
-generator/G_MODEL/A/LayerNorm_2/batchnorm/mulMul/generator/G_MODEL/A/LayerNorm_2/batchnorm/Rsqrt*generator/G_MODEL/A/LayerNorm_2/gamma/read*
T0*&
_output_shapes
:@
Ë
/generator/G_MODEL/A/LayerNorm_2/batchnorm/mul_1Mul!generator/G_MODEL/A/Conv_2/Conv2D-generator/G_MODEL/A/LayerNorm_2/batchnorm/mul*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
T0
Ä
/generator/G_MODEL/A/LayerNorm_2/batchnorm/mul_2Mul,generator/G_MODEL/A/LayerNorm_2/moments/mean-generator/G_MODEL/A/LayerNorm_2/batchnorm/mul*
T0*&
_output_shapes
:@
Á
-generator/G_MODEL/A/LayerNorm_2/batchnorm/subSub)generator/G_MODEL/A/LayerNorm_2/beta/read/generator/G_MODEL/A/LayerNorm_2/batchnorm/mul_2*
T0*&
_output_shapes
:@
Û
/generator/G_MODEL/A/LayerNorm_2/batchnorm/add_1AddV2/generator/G_MODEL/A/LayerNorm_2/batchnorm/mul_1-generator/G_MODEL/A/LayerNorm_2/batchnorm/sub*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
T0
°
generator/G_MODEL/A/LeakyRelu_2	LeakyRelu/generator/G_MODEL/A/LayerNorm_2/batchnorm/add_1*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
alpha%ÍĖL>*
T0

&generator/G_MODEL/B/MirrorPad/paddingsConst*9
value0B."                               *
_output_shapes

:*
dtype0
Ú
generator/G_MODEL/B/MirrorPad	MirrorPadgenerator/G_MODEL/A/LeakyRelu_2&generator/G_MODEL/B/MirrorPad/paddings*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
	Tpaddings0*
T0*
mode	REFLECT
Ņ
Cgenerator/G_MODEL/B/Conv/weights/Initializer/truncated_normal/shapeConst*%
valueB"      @      *
_output_shapes
:*
dtype0*3
_class)
'%loc:@generator/G_MODEL/B/Conv/weights
ž
Bgenerator/G_MODEL/B/Conv/weights/Initializer/truncated_normal/meanConst*3
_class)
'%loc:@generator/G_MODEL/B/Conv/weights*
dtype0*
valueB
 *    *
_output_shapes
: 
ū
Dgenerator/G_MODEL/B/Conv/weights/Initializer/truncated_normal/stddevConst*
valueB
 *=*3
_class)
'%loc:@generator/G_MODEL/B/Conv/weights*
dtype0*
_output_shapes
: 
ē
Mgenerator/G_MODEL/B/Conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalCgenerator/G_MODEL/B/Conv/weights/Initializer/truncated_normal/shape*3
_class)
'%loc:@generator/G_MODEL/B/Conv/weights*
dtype0*
T0*'
_output_shapes
:@*
seed2 *

seed 
Ä
Agenerator/G_MODEL/B/Conv/weights/Initializer/truncated_normal/mulMulMgenerator/G_MODEL/B/Conv/weights/Initializer/truncated_normal/TruncatedNormalDgenerator/G_MODEL/B/Conv/weights/Initializer/truncated_normal/stddev*'
_output_shapes
:@*
T0*3
_class)
'%loc:@generator/G_MODEL/B/Conv/weights
ē
=generator/G_MODEL/B/Conv/weights/Initializer/truncated_normalAddAgenerator/G_MODEL/B/Conv/weights/Initializer/truncated_normal/mulBgenerator/G_MODEL/B/Conv/weights/Initializer/truncated_normal/mean*3
_class)
'%loc:@generator/G_MODEL/B/Conv/weights*
T0*'
_output_shapes
:@
Û
 generator/G_MODEL/B/Conv/weights
VariableV2*
shape:@*'
_output_shapes
:@*
dtype0*
	container *3
_class)
'%loc:@generator/G_MODEL/B/Conv/weights*
shared_name 
Ē
'generator/G_MODEL/B/Conv/weights/AssignAssign generator/G_MODEL/B/Conv/weights=generator/G_MODEL/B/Conv/weights/Initializer/truncated_normal*3
_class)
'%loc:@generator/G_MODEL/B/Conv/weights*
validate_shape(*'
_output_shapes
:@*
T0*
use_locking(
š
%generator/G_MODEL/B/Conv/weights/readIdentity generator/G_MODEL/B/Conv/weights*
T0*3
_class)
'%loc:@generator/G_MODEL/B/Conv/weights*'
_output_shapes
:@
w
&generator/G_MODEL/B/Conv/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
ŧ
generator/G_MODEL/B/Conv/Conv2DConv2Dgenerator/G_MODEL/B/MirrorPad%generator/G_MODEL/B/Conv/weights/read*
paddingVALID*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
strides
*
T0*
	dilations
*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
š
4generator/G_MODEL/B/LayerNorm/beta/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:*5
_class+
)'loc:@generator/G_MODEL/B/LayerNorm/beta
Į
"generator/G_MODEL/B/LayerNorm/beta
VariableV2*
dtype0*
shared_name *5
_class+
)'loc:@generator/G_MODEL/B/LayerNorm/beta*
shape:*
	container *
_output_shapes	
:

)generator/G_MODEL/B/LayerNorm/beta/AssignAssign"generator/G_MODEL/B/LayerNorm/beta4generator/G_MODEL/B/LayerNorm/beta/Initializer/zeros*
use_locking(*
_output_shapes	
:*5
_class+
)'loc:@generator/G_MODEL/B/LayerNorm/beta*
T0*
validate_shape(
ī
'generator/G_MODEL/B/LayerNorm/beta/readIdentity"generator/G_MODEL/B/LayerNorm/beta*
_output_shapes	
:*5
_class+
)'loc:@generator/G_MODEL/B/LayerNorm/beta*
T0
ŧ
4generator/G_MODEL/B/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*
_output_shapes	
:*6
_class,
*(loc:@generator/G_MODEL/B/LayerNorm/gamma*
dtype0
É
#generator/G_MODEL/B/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
	container *6
_class,
*(loc:@generator/G_MODEL/B/LayerNorm/gamma*
shape:

*generator/G_MODEL/B/LayerNorm/gamma/AssignAssign#generator/G_MODEL/B/LayerNorm/gamma4generator/G_MODEL/B/LayerNorm/gamma/Initializer/ones*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@generator/G_MODEL/B/LayerNorm/gamma
·
(generator/G_MODEL/B/LayerNorm/gamma/readIdentity#generator/G_MODEL/B/LayerNorm/gamma*6
_class,
*(loc:@generator/G_MODEL/B/LayerNorm/gamma*
_output_shapes	
:*
T0

<generator/G_MODEL/B/LayerNorm/moments/mean/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0
ß
*generator/G_MODEL/B/LayerNorm/moments/meanMeangenerator/G_MODEL/B/Conv/Conv2D<generator/G_MODEL/B/LayerNorm/moments/mean/reduction_indices*
T0*&
_output_shapes
:*
	keep_dims(*

Tidx0

2generator/G_MODEL/B/LayerNorm/moments/StopGradientStopGradient*generator/G_MODEL/B/LayerNorm/moments/mean*
T0*&
_output_shapes
:
å
7generator/G_MODEL/B/LayerNorm/moments/SquaredDifferenceSquaredDifferencegenerator/G_MODEL/B/Conv/Conv2D2generator/G_MODEL/B/LayerNorm/moments/StopGradient*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ

@generator/G_MODEL/B/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
ĸ
.generator/G_MODEL/B/LayerNorm/moments/varianceMean7generator/G_MODEL/B/LayerNorm/moments/SquaredDifference@generator/G_MODEL/B/LayerNorm/moments/variance/reduction_indices*

Tidx0*&
_output_shapes
:*
T0*
	keep_dims(
r
-generator/G_MODEL/B/LayerNorm/batchnorm/add/yConst*
dtype0*
valueB
 *Ėž+*
_output_shapes
: 
Ä
+generator/G_MODEL/B/LayerNorm/batchnorm/addAddV2.generator/G_MODEL/B/LayerNorm/moments/variance-generator/G_MODEL/B/LayerNorm/batchnorm/add/y*&
_output_shapes
:*
T0

-generator/G_MODEL/B/LayerNorm/batchnorm/RsqrtRsqrt+generator/G_MODEL/B/LayerNorm/batchnorm/add*
T0*&
_output_shapes
:
―
+generator/G_MODEL/B/LayerNorm/batchnorm/mulMul-generator/G_MODEL/B/LayerNorm/batchnorm/Rsqrt(generator/G_MODEL/B/LayerNorm/gamma/read*'
_output_shapes
:*
T0
Æ
-generator/G_MODEL/B/LayerNorm/batchnorm/mul_1Mulgenerator/G_MODEL/B/Conv/Conv2D+generator/G_MODEL/B/LayerNorm/batchnorm/mul*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
ŋ
-generator/G_MODEL/B/LayerNorm/batchnorm/mul_2Mul*generator/G_MODEL/B/LayerNorm/moments/mean+generator/G_MODEL/B/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:
ž
+generator/G_MODEL/B/LayerNorm/batchnorm/subSub'generator/G_MODEL/B/LayerNorm/beta/read-generator/G_MODEL/B/LayerNorm/batchnorm/mul_2*'
_output_shapes
:*
T0
Ö
-generator/G_MODEL/B/LayerNorm/batchnorm/add_1AddV2-generator/G_MODEL/B/LayerNorm/batchnorm/mul_1+generator/G_MODEL/B/LayerNorm/batchnorm/sub*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
­
generator/G_MODEL/B/LeakyRelu	LeakyRelu-generator/G_MODEL/B/LayerNorm/batchnorm/add_1*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
alpha%ÍĖL>*
T0

(generator/G_MODEL/B/MirrorPad_1/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
Ý
generator/G_MODEL/B/MirrorPad_1	MirrorPadgenerator/G_MODEL/B/LeakyRelu(generator/G_MODEL/B/MirrorPad_1/paddings*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0*
	Tpaddings0*
mode	REFLECT
Õ
Egenerator/G_MODEL/B/Conv_1/weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0*5
_class+
)'loc:@generator/G_MODEL/B/Conv_1/weights
Ā
Dgenerator/G_MODEL/B/Conv_1/weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: *5
_class+
)'loc:@generator/G_MODEL/B/Conv_1/weights
Â
Fgenerator/G_MODEL/B/Conv_1/weights/Initializer/truncated_normal/stddevConst*
valueB
 *B=*
_output_shapes
: *
dtype0*5
_class+
)'loc:@generator/G_MODEL/B/Conv_1/weights
đ
Ogenerator/G_MODEL/B/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalEgenerator/G_MODEL/B/Conv_1/weights/Initializer/truncated_normal/shape*(
_output_shapes
:*5
_class+
)'loc:@generator/G_MODEL/B/Conv_1/weights*
dtype0*

seed *
T0*
seed2 
Í
Cgenerator/G_MODEL/B/Conv_1/weights/Initializer/truncated_normal/mulMulOgenerator/G_MODEL/B/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalFgenerator/G_MODEL/B/Conv_1/weights/Initializer/truncated_normal/stddev*
T0*5
_class+
)'loc:@generator/G_MODEL/B/Conv_1/weights*(
_output_shapes
:
ŧ
?generator/G_MODEL/B/Conv_1/weights/Initializer/truncated_normalAddCgenerator/G_MODEL/B/Conv_1/weights/Initializer/truncated_normal/mulDgenerator/G_MODEL/B/Conv_1/weights/Initializer/truncated_normal/mean*
T0*5
_class+
)'loc:@generator/G_MODEL/B/Conv_1/weights*(
_output_shapes
:
á
"generator/G_MODEL/B/Conv_1/weights
VariableV2*
	container *
dtype0*(
_output_shapes
:*5
_class+
)'loc:@generator/G_MODEL/B/Conv_1/weights*
shared_name *
shape:
Ŧ
)generator/G_MODEL/B/Conv_1/weights/AssignAssign"generator/G_MODEL/B/Conv_1/weights?generator/G_MODEL/B/Conv_1/weights/Initializer/truncated_normal*(
_output_shapes
:*
use_locking(*5
_class+
)'loc:@generator/G_MODEL/B/Conv_1/weights*
T0*
validate_shape(
Á
'generator/G_MODEL/B/Conv_1/weights/readIdentity"generator/G_MODEL/B/Conv_1/weights*(
_output_shapes
:*
T0*5
_class+
)'loc:@generator/G_MODEL/B/Conv_1/weights
y
(generator/G_MODEL/B/Conv_1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
Á
!generator/G_MODEL/B/Conv_1/Conv2DConv2Dgenerator/G_MODEL/B/MirrorPad_1'generator/G_MODEL/B/Conv_1/weights/read*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*
	dilations
*
data_formatNHWC*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
ū
6generator/G_MODEL/B/LayerNorm_1/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *7
_class-
+)loc:@generator/G_MODEL/B/LayerNorm_1/beta
Ë
$generator/G_MODEL/B/LayerNorm_1/beta
VariableV2*
shape:*
shared_name *
	container *7
_class-
+)loc:@generator/G_MODEL/B/LayerNorm_1/beta*
dtype0*
_output_shapes	
:

+generator/G_MODEL/B/LayerNorm_1/beta/AssignAssign$generator/G_MODEL/B/LayerNorm_1/beta6generator/G_MODEL/B/LayerNorm_1/beta/Initializer/zeros*
_output_shapes	
:*
T0*7
_class-
+)loc:@generator/G_MODEL/B/LayerNorm_1/beta*
validate_shape(*
use_locking(
š
)generator/G_MODEL/B/LayerNorm_1/beta/readIdentity$generator/G_MODEL/B/LayerNorm_1/beta*
T0*
_output_shapes	
:*7
_class-
+)loc:@generator/G_MODEL/B/LayerNorm_1/beta
ŋ
6generator/G_MODEL/B/LayerNorm_1/gamma/Initializer/onesConst*
valueB*  ?*
_output_shapes	
:*
dtype0*8
_class.
,*loc:@generator/G_MODEL/B/LayerNorm_1/gamma
Í
%generator/G_MODEL/B/LayerNorm_1/gamma
VariableV2*
shared_name *
shape:*
dtype0*8
_class.
,*loc:@generator/G_MODEL/B/LayerNorm_1/gamma*
	container *
_output_shapes	
:

,generator/G_MODEL/B/LayerNorm_1/gamma/AssignAssign%generator/G_MODEL/B/LayerNorm_1/gamma6generator/G_MODEL/B/LayerNorm_1/gamma/Initializer/ones*
_output_shapes	
:*8
_class.
,*loc:@generator/G_MODEL/B/LayerNorm_1/gamma*
validate_shape(*
use_locking(*
T0
―
*generator/G_MODEL/B/LayerNorm_1/gamma/readIdentity%generator/G_MODEL/B/LayerNorm_1/gamma*
T0*
_output_shapes	
:*8
_class.
,*loc:@generator/G_MODEL/B/LayerNorm_1/gamma

>generator/G_MODEL/B/LayerNorm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0
å
,generator/G_MODEL/B/LayerNorm_1/moments/meanMean!generator/G_MODEL/B/Conv_1/Conv2D>generator/G_MODEL/B/LayerNorm_1/moments/mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*&
_output_shapes
:
Ģ
4generator/G_MODEL/B/LayerNorm_1/moments/StopGradientStopGradient,generator/G_MODEL/B/LayerNorm_1/moments/mean*&
_output_shapes
:*
T0
ë
9generator/G_MODEL/B/LayerNorm_1/moments/SquaredDifferenceSquaredDifference!generator/G_MODEL/B/Conv_1/Conv2D4generator/G_MODEL/B/LayerNorm_1/moments/StopGradient*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0

Bgenerator/G_MODEL/B/LayerNorm_1/moments/variance/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0

0generator/G_MODEL/B/LayerNorm_1/moments/varianceMean9generator/G_MODEL/B/LayerNorm_1/moments/SquaredDifferenceBgenerator/G_MODEL/B/LayerNorm_1/moments/variance/reduction_indices*
T0*

Tidx0*&
_output_shapes
:*
	keep_dims(
t
/generator/G_MODEL/B/LayerNorm_1/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *Ėž+*
dtype0
Ę
-generator/G_MODEL/B/LayerNorm_1/batchnorm/addAddV20generator/G_MODEL/B/LayerNorm_1/moments/variance/generator/G_MODEL/B/LayerNorm_1/batchnorm/add/y*&
_output_shapes
:*
T0

/generator/G_MODEL/B/LayerNorm_1/batchnorm/RsqrtRsqrt-generator/G_MODEL/B/LayerNorm_1/batchnorm/add*
T0*&
_output_shapes
:
Ã
-generator/G_MODEL/B/LayerNorm_1/batchnorm/mulMul/generator/G_MODEL/B/LayerNorm_1/batchnorm/Rsqrt*generator/G_MODEL/B/LayerNorm_1/gamma/read*
T0*'
_output_shapes
:
Ė
/generator/G_MODEL/B/LayerNorm_1/batchnorm/mul_1Mul!generator/G_MODEL/B/Conv_1/Conv2D-generator/G_MODEL/B/LayerNorm_1/batchnorm/mul*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Å
/generator/G_MODEL/B/LayerNorm_1/batchnorm/mul_2Mul,generator/G_MODEL/B/LayerNorm_1/moments/mean-generator/G_MODEL/B/LayerNorm_1/batchnorm/mul*'
_output_shapes
:*
T0
Â
-generator/G_MODEL/B/LayerNorm_1/batchnorm/subSub)generator/G_MODEL/B/LayerNorm_1/beta/read/generator/G_MODEL/B/LayerNorm_1/batchnorm/mul_2*'
_output_shapes
:*
T0
Ü
/generator/G_MODEL/B/LayerNorm_1/batchnorm/add_1AddV2/generator/G_MODEL/B/LayerNorm_1/batchnorm/mul_1-generator/G_MODEL/B/LayerNorm_1/batchnorm/sub*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
ą
generator/G_MODEL/B/LeakyRelu_1	LeakyRelu/generator/G_MODEL/B/LayerNorm_1/batchnorm/add_1*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0*
alpha%ÍĖL>

&generator/G_MODEL/C/MirrorPad/paddingsConst*
dtype0*9
value0B."                             *
_output_shapes

:
Û
generator/G_MODEL/C/MirrorPad	MirrorPadgenerator/G_MODEL/B/LeakyRelu_1&generator/G_MODEL/C/MirrorPad/paddings*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
	Tpaddings0*
mode	REFLECT
Ņ
Cgenerator/G_MODEL/C/Conv/weights/Initializer/truncated_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:*3
_class)
'%loc:@generator/G_MODEL/C/Conv/weights
ž
Bgenerator/G_MODEL/C/Conv/weights/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    *3
_class)
'%loc:@generator/G_MODEL/C/Conv/weights
ū
Dgenerator/G_MODEL/C/Conv/weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*3
_class)
'%loc:@generator/G_MODEL/C/Conv/weights*
valueB
 *B=
ģ
Mgenerator/G_MODEL/C/Conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalCgenerator/G_MODEL/C/Conv/weights/Initializer/truncated_normal/shape*
dtype0*3
_class)
'%loc:@generator/G_MODEL/C/Conv/weights*
seed2 *
T0*(
_output_shapes
:*

seed 
Å
Agenerator/G_MODEL/C/Conv/weights/Initializer/truncated_normal/mulMulMgenerator/G_MODEL/C/Conv/weights/Initializer/truncated_normal/TruncatedNormalDgenerator/G_MODEL/C/Conv/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:*
T0*3
_class)
'%loc:@generator/G_MODEL/C/Conv/weights
ģ
=generator/G_MODEL/C/Conv/weights/Initializer/truncated_normalAddAgenerator/G_MODEL/C/Conv/weights/Initializer/truncated_normal/mulBgenerator/G_MODEL/C/Conv/weights/Initializer/truncated_normal/mean*(
_output_shapes
:*3
_class)
'%loc:@generator/G_MODEL/C/Conv/weights*
T0
Ý
 generator/G_MODEL/C/Conv/weights
VariableV2*(
_output_shapes
:*3
_class)
'%loc:@generator/G_MODEL/C/Conv/weights*
shape:*
dtype0*
	container *
shared_name 
Ģ
'generator/G_MODEL/C/Conv/weights/AssignAssign generator/G_MODEL/C/Conv/weights=generator/G_MODEL/C/Conv/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:*3
_class)
'%loc:@generator/G_MODEL/C/Conv/weights*
T0*
use_locking(
ŧ
%generator/G_MODEL/C/Conv/weights/readIdentity generator/G_MODEL/C/Conv/weights*(
_output_shapes
:*3
_class)
'%loc:@generator/G_MODEL/C/Conv/weights*
T0
w
&generator/G_MODEL/C/Conv/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ŧ
generator/G_MODEL/C/Conv/Conv2DConv2Dgenerator/G_MODEL/C/MirrorPad%generator/G_MODEL/C/Conv/weights/read*
T0*
strides
*
paddingVALID*
use_cudnn_on_gpu(*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
explicit_paddings
 *
data_formatNHWC*
	dilations

š
4generator/G_MODEL/C/LayerNorm/beta/Initializer/zerosConst*5
_class+
)'loc:@generator/G_MODEL/C/LayerNorm/beta*
dtype0*
valueB*    *
_output_shapes	
:
Į
"generator/G_MODEL/C/LayerNorm/beta
VariableV2*
shape:*5
_class+
)'loc:@generator/G_MODEL/C/LayerNorm/beta*
	container *
_output_shapes	
:*
dtype0*
shared_name 

)generator/G_MODEL/C/LayerNorm/beta/AssignAssign"generator/G_MODEL/C/LayerNorm/beta4generator/G_MODEL/C/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@generator/G_MODEL/C/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ī
'generator/G_MODEL/C/LayerNorm/beta/readIdentity"generator/G_MODEL/C/LayerNorm/beta*
T0*5
_class+
)'loc:@generator/G_MODEL/C/LayerNorm/beta*
_output_shapes	
:
ŧ
4generator/G_MODEL/C/LayerNorm/gamma/Initializer/onesConst*
dtype0*6
_class,
*(loc:@generator/G_MODEL/C/LayerNorm/gamma*
_output_shapes	
:*
valueB*  ?
É
#generator/G_MODEL/C/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:*6
_class,
*(loc:@generator/G_MODEL/C/LayerNorm/gamma*
shared_name *
shape:*
	container 

*generator/G_MODEL/C/LayerNorm/gamma/AssignAssign#generator/G_MODEL/C/LayerNorm/gamma4generator/G_MODEL/C/LayerNorm/gamma/Initializer/ones*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(*6
_class,
*(loc:@generator/G_MODEL/C/LayerNorm/gamma
·
(generator/G_MODEL/C/LayerNorm/gamma/readIdentity#generator/G_MODEL/C/LayerNorm/gamma*6
_class,
*(loc:@generator/G_MODEL/C/LayerNorm/gamma*
_output_shapes	
:*
T0

<generator/G_MODEL/C/LayerNorm/moments/mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:
ß
*generator/G_MODEL/C/LayerNorm/moments/meanMeangenerator/G_MODEL/C/Conv/Conv2D<generator/G_MODEL/C/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*&
_output_shapes
:*
T0*

Tidx0

2generator/G_MODEL/C/LayerNorm/moments/StopGradientStopGradient*generator/G_MODEL/C/LayerNorm/moments/mean*
T0*&
_output_shapes
:
å
7generator/G_MODEL/C/LayerNorm/moments/SquaredDifferenceSquaredDifferencegenerator/G_MODEL/C/Conv/Conv2D2generator/G_MODEL/C/LayerNorm/moments/StopGradient*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0

@generator/G_MODEL/C/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
ĸ
.generator/G_MODEL/C/LayerNorm/moments/varianceMean7generator/G_MODEL/C/LayerNorm/moments/SquaredDifference@generator/G_MODEL/C/LayerNorm/moments/variance/reduction_indices*

Tidx0*&
_output_shapes
:*
	keep_dims(*
T0
r
-generator/G_MODEL/C/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ėž+
Ä
+generator/G_MODEL/C/LayerNorm/batchnorm/addAddV2.generator/G_MODEL/C/LayerNorm/moments/variance-generator/G_MODEL/C/LayerNorm/batchnorm/add/y*&
_output_shapes
:*
T0

-generator/G_MODEL/C/LayerNorm/batchnorm/RsqrtRsqrt+generator/G_MODEL/C/LayerNorm/batchnorm/add*
T0*&
_output_shapes
:
―
+generator/G_MODEL/C/LayerNorm/batchnorm/mulMul-generator/G_MODEL/C/LayerNorm/batchnorm/Rsqrt(generator/G_MODEL/C/LayerNorm/gamma/read*'
_output_shapes
:*
T0
Æ
-generator/G_MODEL/C/LayerNorm/batchnorm/mul_1Mulgenerator/G_MODEL/C/Conv/Conv2D+generator/G_MODEL/C/LayerNorm/batchnorm/mul*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
ŋ
-generator/G_MODEL/C/LayerNorm/batchnorm/mul_2Mul*generator/G_MODEL/C/LayerNorm/moments/mean+generator/G_MODEL/C/LayerNorm/batchnorm/mul*'
_output_shapes
:*
T0
ž
+generator/G_MODEL/C/LayerNorm/batchnorm/subSub'generator/G_MODEL/C/LayerNorm/beta/read-generator/G_MODEL/C/LayerNorm/batchnorm/mul_2*'
_output_shapes
:*
T0
Ö
-generator/G_MODEL/C/LayerNorm/batchnorm/add_1AddV2-generator/G_MODEL/C/LayerNorm/batchnorm/mul_1+generator/G_MODEL/C/LayerNorm/batchnorm/sub*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
­
generator/G_MODEL/C/LeakyRelu	LeakyRelu-generator/G_MODEL/C/LayerNorm/batchnorm/add_1*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0*
alpha%ÍĖL>
Ũ
Fgenerator/G_MODEL/C/r1/Conv/weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0*6
_class,
*(loc:@generator/G_MODEL/C/r1/Conv/weights
Â
Egenerator/G_MODEL/C/r1/Conv/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: *6
_class,
*(loc:@generator/G_MODEL/C/r1/Conv/weights
Ä
Ggenerator/G_MODEL/C/r1/Conv/weights/Initializer/truncated_normal/stddevConst*6
_class,
*(loc:@generator/G_MODEL/C/r1/Conv/weights*
dtype0*
_output_shapes
: *
valueB
 *Eņ>
ž
Pgenerator/G_MODEL/C/r1/Conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalFgenerator/G_MODEL/C/r1/Conv/weights/Initializer/truncated_normal/shape*
T0*
seed2 *6
_class,
*(loc:@generator/G_MODEL/C/r1/Conv/weights*(
_output_shapes
:*

seed *
dtype0
Ņ
Dgenerator/G_MODEL/C/r1/Conv/weights/Initializer/truncated_normal/mulMulPgenerator/G_MODEL/C/r1/Conv/weights/Initializer/truncated_normal/TruncatedNormalGgenerator/G_MODEL/C/r1/Conv/weights/Initializer/truncated_normal/stddev*6
_class,
*(loc:@generator/G_MODEL/C/r1/Conv/weights*
T0*(
_output_shapes
:
ŋ
@generator/G_MODEL/C/r1/Conv/weights/Initializer/truncated_normalAddDgenerator/G_MODEL/C/r1/Conv/weights/Initializer/truncated_normal/mulEgenerator/G_MODEL/C/r1/Conv/weights/Initializer/truncated_normal/mean*(
_output_shapes
:*6
_class,
*(loc:@generator/G_MODEL/C/r1/Conv/weights*
T0
ã
#generator/G_MODEL/C/r1/Conv/weights
VariableV2*
	container *6
_class,
*(loc:@generator/G_MODEL/C/r1/Conv/weights*
dtype0*
shared_name *
shape:*(
_output_shapes
:
Ŋ
*generator/G_MODEL/C/r1/Conv/weights/AssignAssign#generator/G_MODEL/C/r1/Conv/weights@generator/G_MODEL/C/r1/Conv/weights/Initializer/truncated_normal*6
_class,
*(loc:@generator/G_MODEL/C/r1/Conv/weights*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Ä
(generator/G_MODEL/C/r1/Conv/weights/readIdentity#generator/G_MODEL/C/r1/Conv/weights*6
_class,
*(loc:@generator/G_MODEL/C/r1/Conv/weights*
T0*(
_output_shapes
:
z
)generator/G_MODEL/C/r1/Conv/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
Á
"generator/G_MODEL/C/r1/Conv/Conv2DConv2Dgenerator/G_MODEL/C/LeakyRelu(generator/G_MODEL/C/r1/Conv/weights/read*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
	dilations
*
strides

Ā
7generator/G_MODEL/C/r1/LayerNorm/beta/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:*8
_class.
,*loc:@generator/G_MODEL/C/r1/LayerNorm/beta
Í
%generator/G_MODEL/C/r1/LayerNorm/beta
VariableV2*8
_class.
,*loc:@generator/G_MODEL/C/r1/LayerNorm/beta*
_output_shapes	
:*
shared_name *
	container *
dtype0*
shape:

,generator/G_MODEL/C/r1/LayerNorm/beta/AssignAssign%generator/G_MODEL/C/r1/LayerNorm/beta7generator/G_MODEL/C/r1/LayerNorm/beta/Initializer/zeros*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(*8
_class.
,*loc:@generator/G_MODEL/C/r1/LayerNorm/beta
―
*generator/G_MODEL/C/r1/LayerNorm/beta/readIdentity%generator/G_MODEL/C/r1/LayerNorm/beta*
_output_shapes	
:*8
_class.
,*loc:@generator/G_MODEL/C/r1/LayerNorm/beta*
T0
Á
7generator/G_MODEL/C/r1/LayerNorm/gamma/Initializer/onesConst*9
_class/
-+loc:@generator/G_MODEL/C/r1/LayerNorm/gamma*
_output_shapes	
:*
valueB*  ?*
dtype0
Ï
&generator/G_MODEL/C/r1/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:*9
_class/
-+loc:@generator/G_MODEL/C/r1/LayerNorm/gamma*
	container *
shared_name *
shape:
Ē
-generator/G_MODEL/C/r1/LayerNorm/gamma/AssignAssign&generator/G_MODEL/C/r1/LayerNorm/gamma7generator/G_MODEL/C/r1/LayerNorm/gamma/Initializer/ones*9
_class/
-+loc:@generator/G_MODEL/C/r1/LayerNorm/gamma*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0
Ā
+generator/G_MODEL/C/r1/LayerNorm/gamma/readIdentity&generator/G_MODEL/C/r1/LayerNorm/gamma*
T0*9
_class/
-+loc:@generator/G_MODEL/C/r1/LayerNorm/gamma*
_output_shapes	
:

?generator/G_MODEL/C/r1/LayerNorm/moments/mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:
č
-generator/G_MODEL/C/r1/LayerNorm/moments/meanMean"generator/G_MODEL/C/r1/Conv/Conv2D?generator/G_MODEL/C/r1/LayerNorm/moments/mean/reduction_indices*
T0*
	keep_dims(*

Tidx0*&
_output_shapes
:
Ĩ
5generator/G_MODEL/C/r1/LayerNorm/moments/StopGradientStopGradient-generator/G_MODEL/C/r1/LayerNorm/moments/mean*&
_output_shapes
:*
T0
î
:generator/G_MODEL/C/r1/LayerNorm/moments/SquaredDifferenceSquaredDifference"generator/G_MODEL/C/r1/Conv/Conv2D5generator/G_MODEL/C/r1/LayerNorm/moments/StopGradient*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ

Cgenerator/G_MODEL/C/r1/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         

1generator/G_MODEL/C/r1/LayerNorm/moments/varianceMean:generator/G_MODEL/C/r1/LayerNorm/moments/SquaredDifferenceCgenerator/G_MODEL/C/r1/LayerNorm/moments/variance/reduction_indices*

Tidx0*
T0*&
_output_shapes
:*
	keep_dims(
u
0generator/G_MODEL/C/r1/LayerNorm/batchnorm/add/yConst*
valueB
 *Ėž+*
_output_shapes
: *
dtype0
Í
.generator/G_MODEL/C/r1/LayerNorm/batchnorm/addAddV21generator/G_MODEL/C/r1/LayerNorm/moments/variance0generator/G_MODEL/C/r1/LayerNorm/batchnorm/add/y*
T0*&
_output_shapes
:

0generator/G_MODEL/C/r1/LayerNorm/batchnorm/RsqrtRsqrt.generator/G_MODEL/C/r1/LayerNorm/batchnorm/add*&
_output_shapes
:*
T0
Æ
.generator/G_MODEL/C/r1/LayerNorm/batchnorm/mulMul0generator/G_MODEL/C/r1/LayerNorm/batchnorm/Rsqrt+generator/G_MODEL/C/r1/LayerNorm/gamma/read*
T0*'
_output_shapes
:
Ï
0generator/G_MODEL/C/r1/LayerNorm/batchnorm/mul_1Mul"generator/G_MODEL/C/r1/Conv/Conv2D.generator/G_MODEL/C/r1/LayerNorm/batchnorm/mul*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
Č
0generator/G_MODEL/C/r1/LayerNorm/batchnorm/mul_2Mul-generator/G_MODEL/C/r1/LayerNorm/moments/mean.generator/G_MODEL/C/r1/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:
Å
.generator/G_MODEL/C/r1/LayerNorm/batchnorm/subSub*generator/G_MODEL/C/r1/LayerNorm/beta/read0generator/G_MODEL/C/r1/LayerNorm/batchnorm/mul_2*'
_output_shapes
:*
T0
ß
0generator/G_MODEL/C/r1/LayerNorm/batchnorm/add_1AddV20generator/G_MODEL/C/r1/LayerNorm/batchnorm/mul_1.generator/G_MODEL/C/r1/LayerNorm/batchnorm/sub*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
ģ
 generator/G_MODEL/C/r1/LeakyRelu	LeakyRelu0generator/G_MODEL/C/r1/LayerNorm/batchnorm/add_1*
alpha%ÍĖL>*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ

)generator/G_MODEL/C/r1/MirrorPad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
â
 generator/G_MODEL/C/r1/MirrorPad	MirrorPad generator/G_MODEL/C/r1/LeakyRelu)generator/G_MODEL/C/r1/MirrorPad/paddings*
T0*
	Tpaddings0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
mode	REFLECT
Į
>generator/G_MODEL/C/r1/r1/w/Initializer/truncated_normal/shapeConst*%
valueB"            *.
_class$
" loc:@generator/G_MODEL/C/r1/r1/w*
_output_shapes
:*
dtype0
ē
=generator/G_MODEL/C/r1/r1/w/Initializer/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0*.
_class$
" loc:@generator/G_MODEL/C/r1/r1/w
ī
?generator/G_MODEL/C/r1/r1/w/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *	=*
dtype0*.
_class$
" loc:@generator/G_MODEL/C/r1/r1/w
Ģ
Hgenerator/G_MODEL/C/r1/r1/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>generator/G_MODEL/C/r1/r1/w/Initializer/truncated_normal/shape*
seed2 *

seed *
dtype0*.
_class$
" loc:@generator/G_MODEL/C/r1/r1/w*'
_output_shapes
:*
T0
°
<generator/G_MODEL/C/r1/r1/w/Initializer/truncated_normal/mulMulHgenerator/G_MODEL/C/r1/r1/w/Initializer/truncated_normal/TruncatedNormal?generator/G_MODEL/C/r1/r1/w/Initializer/truncated_normal/stddev*'
_output_shapes
:*
T0*.
_class$
" loc:@generator/G_MODEL/C/r1/r1/w

8generator/G_MODEL/C/r1/r1/w/Initializer/truncated_normalAdd<generator/G_MODEL/C/r1/r1/w/Initializer/truncated_normal/mul=generator/G_MODEL/C/r1/r1/w/Initializer/truncated_normal/mean*'
_output_shapes
:*.
_class$
" loc:@generator/G_MODEL/C/r1/r1/w*
T0
Ņ
generator/G_MODEL/C/r1/r1/w
VariableV2*
shape:*'
_output_shapes
:*
dtype0*
	container *
shared_name *.
_class$
" loc:@generator/G_MODEL/C/r1/r1/w

"generator/G_MODEL/C/r1/r1/w/AssignAssigngenerator/G_MODEL/C/r1/r1/w8generator/G_MODEL/C/r1/r1/w/Initializer/truncated_normal*.
_class$
" loc:@generator/G_MODEL/C/r1/r1/w*
use_locking(*
validate_shape(*'
_output_shapes
:*
T0
Ŧ
 generator/G_MODEL/C/r1/r1/w/readIdentitygenerator/G_MODEL/C/r1/r1/w*
T0*.
_class$
" loc:@generator/G_MODEL/C/r1/r1/w*'
_output_shapes
:
{
"generator/G_MODEL/C/r1/r1/r1/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
{
*generator/G_MODEL/C/r1/r1/r1/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0

generator/G_MODEL/C/r1/r1/r1DepthwiseConv2dNative generator/G_MODEL/C/r1/MirrorPad generator/G_MODEL/C/r1/r1/w/read*
	dilations
*
paddingVALID*
strides
*
T0*
data_formatNHWC*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
ē
0generator/G_MODEL/C/r1/r1/bias/Initializer/ConstConst*
_output_shapes	
:*1
_class'
%#loc:@generator/G_MODEL/C/r1/r1/bias*
dtype0*
valueB*    
ŋ
generator/G_MODEL/C/r1/r1/bias
VariableV2*
_output_shapes	
:*
dtype0*
shape:*
shared_name *
	container *1
_class'
%#loc:@generator/G_MODEL/C/r1/r1/bias

%generator/G_MODEL/C/r1/r1/bias/AssignAssigngenerator/G_MODEL/C/r1/r1/bias0generator/G_MODEL/C/r1/r1/bias/Initializer/Const*
use_locking(*
validate_shape(*1
_class'
%#loc:@generator/G_MODEL/C/r1/r1/bias*
_output_shapes	
:*
T0
Ļ
#generator/G_MODEL/C/r1/r1/bias/readIdentitygenerator/G_MODEL/C/r1/r1/bias*
T0*
_output_shapes	
:*1
_class'
%#loc:@generator/G_MODEL/C/r1/r1/bias
Ę
!generator/G_MODEL/C/r1/r1/BiasAddBiasAddgenerator/G_MODEL/C/r1/r1/r1#generator/G_MODEL/C/r1/r1/bias/read*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0*
data_formatNHWC
°
/generator/G_MODEL/C/r1/1/beta/Initializer/zerosConst*0
_class&
$"loc:@generator/G_MODEL/C/r1/1/beta*
_output_shapes	
:*
valueB*    *
dtype0
―
generator/G_MODEL/C/r1/1/beta
VariableV2*0
_class&
$"loc:@generator/G_MODEL/C/r1/1/beta*
	container *
shared_name *
shape:*
_output_shapes	
:*
dtype0
ĸ
$generator/G_MODEL/C/r1/1/beta/AssignAssigngenerator/G_MODEL/C/r1/1/beta/generator/G_MODEL/C/r1/1/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*0
_class&
$"loc:@generator/G_MODEL/C/r1/1/beta*
T0
Ĩ
"generator/G_MODEL/C/r1/1/beta/readIdentitygenerator/G_MODEL/C/r1/1/beta*
T0*0
_class&
$"loc:@generator/G_MODEL/C/r1/1/beta*
_output_shapes	
:
ą
/generator/G_MODEL/C/r1/1/gamma/Initializer/onesConst*
valueB*  ?*1
_class'
%#loc:@generator/G_MODEL/C/r1/1/gamma*
dtype0*
_output_shapes	
:
ŋ
generator/G_MODEL/C/r1/1/gamma
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:*1
_class'
%#loc:@generator/G_MODEL/C/r1/1/gamma

%generator/G_MODEL/C/r1/1/gamma/AssignAssigngenerator/G_MODEL/C/r1/1/gamma/generator/G_MODEL/C/r1/1/gamma/Initializer/ones*1
_class'
%#loc:@generator/G_MODEL/C/r1/1/gamma*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(
Ļ
#generator/G_MODEL/C/r1/1/gamma/readIdentitygenerator/G_MODEL/C/r1/1/gamma*1
_class'
%#loc:@generator/G_MODEL/C/r1/1/gamma*
_output_shapes	
:*
T0

7generator/G_MODEL/C/r1/1/moments/mean/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0
Ũ
%generator/G_MODEL/C/r1/1/moments/meanMean!generator/G_MODEL/C/r1/r1/BiasAdd7generator/G_MODEL/C/r1/1/moments/mean/reduction_indices*&
_output_shapes
:*
	keep_dims(*

Tidx0*
T0

-generator/G_MODEL/C/r1/1/moments/StopGradientStopGradient%generator/G_MODEL/C/r1/1/moments/mean*
T0*&
_output_shapes
:
Ý
2generator/G_MODEL/C/r1/1/moments/SquaredDifferenceSquaredDifference!generator/G_MODEL/C/r1/r1/BiasAdd-generator/G_MODEL/C/r1/1/moments/StopGradient*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ

;generator/G_MODEL/C/r1/1/moments/variance/reduction_indicesConst*
dtype0*!
valueB"         *
_output_shapes
:
ð
)generator/G_MODEL/C/r1/1/moments/varianceMean2generator/G_MODEL/C/r1/1/moments/SquaredDifference;generator/G_MODEL/C/r1/1/moments/variance/reduction_indices*&
_output_shapes
:*

Tidx0*
	keep_dims(*
T0
m
(generator/G_MODEL/C/r1/1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ėž+
ĩ
&generator/G_MODEL/C/r1/1/batchnorm/addAddV2)generator/G_MODEL/C/r1/1/moments/variance(generator/G_MODEL/C/r1/1/batchnorm/add/y*&
_output_shapes
:*
T0

(generator/G_MODEL/C/r1/1/batchnorm/RsqrtRsqrt&generator/G_MODEL/C/r1/1/batchnorm/add*
T0*&
_output_shapes
:
Ū
&generator/G_MODEL/C/r1/1/batchnorm/mulMul(generator/G_MODEL/C/r1/1/batchnorm/Rsqrt#generator/G_MODEL/C/r1/1/gamma/read*'
_output_shapes
:*
T0
ū
(generator/G_MODEL/C/r1/1/batchnorm/mul_1Mul!generator/G_MODEL/C/r1/r1/BiasAdd&generator/G_MODEL/C/r1/1/batchnorm/mul*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
°
(generator/G_MODEL/C/r1/1/batchnorm/mul_2Mul%generator/G_MODEL/C/r1/1/moments/mean&generator/G_MODEL/C/r1/1/batchnorm/mul*'
_output_shapes
:*
T0
­
&generator/G_MODEL/C/r1/1/batchnorm/subSub"generator/G_MODEL/C/r1/1/beta/read(generator/G_MODEL/C/r1/1/batchnorm/mul_2*'
_output_shapes
:*
T0
Į
(generator/G_MODEL/C/r1/1/batchnorm/add_1AddV2(generator/G_MODEL/C/r1/1/batchnorm/mul_1&generator/G_MODEL/C/r1/1/batchnorm/sub*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
­
"generator/G_MODEL/C/r1/LeakyRelu_1	LeakyRelu(generator/G_MODEL/C/r1/1/batchnorm/add_1*
alpha%ÍĖL>*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Û
Hgenerator/G_MODEL/C/r1/Conv_1/weights/Initializer/truncated_normal/shapeConst*8
_class.
,*loc:@generator/G_MODEL/C/r1/Conv_1/weights*
_output_shapes
:*
dtype0*%
valueB"            
Æ
Ggenerator/G_MODEL/C/r1/Conv_1/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0*8
_class.
,*loc:@generator/G_MODEL/C/r1/Conv_1/weights
Č
Igenerator/G_MODEL/C/r1/Conv_1/weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *ÐdÎ=*8
_class.
,*loc:@generator/G_MODEL/C/r1/Conv_1/weights*
dtype0
Â
Rgenerator/G_MODEL/C/r1/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHgenerator/G_MODEL/C/r1/Conv_1/weights/Initializer/truncated_normal/shape*
dtype0*

seed *8
_class.
,*loc:@generator/G_MODEL/C/r1/Conv_1/weights*
seed2 *
T0*(
_output_shapes
:
Ų
Fgenerator/G_MODEL/C/r1/Conv_1/weights/Initializer/truncated_normal/mulMulRgenerator/G_MODEL/C/r1/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalIgenerator/G_MODEL/C/r1/Conv_1/weights/Initializer/truncated_normal/stddev*8
_class.
,*loc:@generator/G_MODEL/C/r1/Conv_1/weights*(
_output_shapes
:*
T0
Į
Bgenerator/G_MODEL/C/r1/Conv_1/weights/Initializer/truncated_normalAddFgenerator/G_MODEL/C/r1/Conv_1/weights/Initializer/truncated_normal/mulGgenerator/G_MODEL/C/r1/Conv_1/weights/Initializer/truncated_normal/mean*8
_class.
,*loc:@generator/G_MODEL/C/r1/Conv_1/weights*(
_output_shapes
:*
T0
į
%generator/G_MODEL/C/r1/Conv_1/weights
VariableV2*
	container *(
_output_shapes
:*
shape:*
dtype0*
shared_name *8
_class.
,*loc:@generator/G_MODEL/C/r1/Conv_1/weights
·
,generator/G_MODEL/C/r1/Conv_1/weights/AssignAssign%generator/G_MODEL/C/r1/Conv_1/weightsBgenerator/G_MODEL/C/r1/Conv_1/weights/Initializer/truncated_normal*8
_class.
,*loc:@generator/G_MODEL/C/r1/Conv_1/weights*
T0*
validate_shape(*(
_output_shapes
:*
use_locking(
Ę
*generator/G_MODEL/C/r1/Conv_1/weights/readIdentity%generator/G_MODEL/C/r1/Conv_1/weights*
T0*8
_class.
,*loc:@generator/G_MODEL/C/r1/Conv_1/weights*(
_output_shapes
:
|
+generator/G_MODEL/C/r1/Conv_1/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ę
$generator/G_MODEL/C/r1/Conv_1/Conv2DConv2D"generator/G_MODEL/C/r1/LeakyRelu_1*generator/G_MODEL/C/r1/Conv_1/weights/read*
	dilations
*
use_cudnn_on_gpu(*
paddingVALID*
explicit_paddings
 *9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0*
data_formatNHWC*
strides

°
/generator/G_MODEL/C/r1/2/beta/Initializer/zerosConst*
dtype0*
valueB*    *0
_class&
$"loc:@generator/G_MODEL/C/r1/2/beta*
_output_shapes	
:
―
generator/G_MODEL/C/r1/2/beta
VariableV2*
shape:*
	container *
shared_name *
_output_shapes	
:*
dtype0*0
_class&
$"loc:@generator/G_MODEL/C/r1/2/beta
ĸ
$generator/G_MODEL/C/r1/2/beta/AssignAssigngenerator/G_MODEL/C/r1/2/beta/generator/G_MODEL/C/r1/2/beta/Initializer/zeros*
_output_shapes	
:*0
_class&
$"loc:@generator/G_MODEL/C/r1/2/beta*
validate_shape(*
use_locking(*
T0
Ĩ
"generator/G_MODEL/C/r1/2/beta/readIdentitygenerator/G_MODEL/C/r1/2/beta*0
_class&
$"loc:@generator/G_MODEL/C/r1/2/beta*
T0*
_output_shapes	
:
ą
/generator/G_MODEL/C/r1/2/gamma/Initializer/onesConst*
valueB*  ?*
_output_shapes	
:*
dtype0*1
_class'
%#loc:@generator/G_MODEL/C/r1/2/gamma
ŋ
generator/G_MODEL/C/r1/2/gamma
VariableV2*
shared_name *
dtype0*
shape:*1
_class'
%#loc:@generator/G_MODEL/C/r1/2/gamma*
	container *
_output_shapes	
:

%generator/G_MODEL/C/r1/2/gamma/AssignAssigngenerator/G_MODEL/C/r1/2/gamma/generator/G_MODEL/C/r1/2/gamma/Initializer/ones*
T0*
_output_shapes	
:*1
_class'
%#loc:@generator/G_MODEL/C/r1/2/gamma*
validate_shape(*
use_locking(
Ļ
#generator/G_MODEL/C/r1/2/gamma/readIdentitygenerator/G_MODEL/C/r1/2/gamma*
T0*
_output_shapes	
:*1
_class'
%#loc:@generator/G_MODEL/C/r1/2/gamma

7generator/G_MODEL/C/r1/2/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0
Ú
%generator/G_MODEL/C/r1/2/moments/meanMean$generator/G_MODEL/C/r1/Conv_1/Conv2D7generator/G_MODEL/C/r1/2/moments/mean/reduction_indices*
T0*&
_output_shapes
:*

Tidx0*
	keep_dims(

-generator/G_MODEL/C/r1/2/moments/StopGradientStopGradient%generator/G_MODEL/C/r1/2/moments/mean*&
_output_shapes
:*
T0
ā
2generator/G_MODEL/C/r1/2/moments/SquaredDifferenceSquaredDifference$generator/G_MODEL/C/r1/Conv_1/Conv2D-generator/G_MODEL/C/r1/2/moments/StopGradient*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0

;generator/G_MODEL/C/r1/2/moments/variance/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0
ð
)generator/G_MODEL/C/r1/2/moments/varianceMean2generator/G_MODEL/C/r1/2/moments/SquaredDifference;generator/G_MODEL/C/r1/2/moments/variance/reduction_indices*
	keep_dims(*
T0*

Tidx0*&
_output_shapes
:
m
(generator/G_MODEL/C/r1/2/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *Ėž+*
dtype0
ĩ
&generator/G_MODEL/C/r1/2/batchnorm/addAddV2)generator/G_MODEL/C/r1/2/moments/variance(generator/G_MODEL/C/r1/2/batchnorm/add/y*&
_output_shapes
:*
T0

(generator/G_MODEL/C/r1/2/batchnorm/RsqrtRsqrt&generator/G_MODEL/C/r1/2/batchnorm/add*&
_output_shapes
:*
T0
Ū
&generator/G_MODEL/C/r1/2/batchnorm/mulMul(generator/G_MODEL/C/r1/2/batchnorm/Rsqrt#generator/G_MODEL/C/r1/2/gamma/read*
T0*'
_output_shapes
:
Á
(generator/G_MODEL/C/r1/2/batchnorm/mul_1Mul$generator/G_MODEL/C/r1/Conv_1/Conv2D&generator/G_MODEL/C/r1/2/batchnorm/mul*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
°
(generator/G_MODEL/C/r1/2/batchnorm/mul_2Mul%generator/G_MODEL/C/r1/2/moments/mean&generator/G_MODEL/C/r1/2/batchnorm/mul*
T0*'
_output_shapes
:
­
&generator/G_MODEL/C/r1/2/batchnorm/subSub"generator/G_MODEL/C/r1/2/beta/read(generator/G_MODEL/C/r1/2/batchnorm/mul_2*
T0*'
_output_shapes
:
Į
(generator/G_MODEL/C/r1/2/batchnorm/add_1AddV2(generator/G_MODEL/C/r1/2/batchnorm/mul_1&generator/G_MODEL/C/r1/2/batchnorm/sub*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
Ũ
Fgenerator/G_MODEL/C/r2/Conv/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:*6
_class,
*(loc:@generator/G_MODEL/C/r2/Conv/weights
Â
Egenerator/G_MODEL/C/r2/Conv/weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: *6
_class,
*(loc:@generator/G_MODEL/C/r2/Conv/weights
Ä
Ggenerator/G_MODEL/C/r2/Conv/weights/Initializer/truncated_normal/stddevConst*
valueB
 *ÐdÎ=*
_output_shapes
: *
dtype0*6
_class,
*(loc:@generator/G_MODEL/C/r2/Conv/weights
ž
Pgenerator/G_MODEL/C/r2/Conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalFgenerator/G_MODEL/C/r2/Conv/weights/Initializer/truncated_normal/shape*(
_output_shapes
:*
dtype0*
T0*

seed *
seed2 *6
_class,
*(loc:@generator/G_MODEL/C/r2/Conv/weights
Ņ
Dgenerator/G_MODEL/C/r2/Conv/weights/Initializer/truncated_normal/mulMulPgenerator/G_MODEL/C/r2/Conv/weights/Initializer/truncated_normal/TruncatedNormalGgenerator/G_MODEL/C/r2/Conv/weights/Initializer/truncated_normal/stddev*
T0*(
_output_shapes
:*6
_class,
*(loc:@generator/G_MODEL/C/r2/Conv/weights
ŋ
@generator/G_MODEL/C/r2/Conv/weights/Initializer/truncated_normalAddDgenerator/G_MODEL/C/r2/Conv/weights/Initializer/truncated_normal/mulEgenerator/G_MODEL/C/r2/Conv/weights/Initializer/truncated_normal/mean*(
_output_shapes
:*
T0*6
_class,
*(loc:@generator/G_MODEL/C/r2/Conv/weights
ã
#generator/G_MODEL/C/r2/Conv/weights
VariableV2*
shape:*
	container *6
_class,
*(loc:@generator/G_MODEL/C/r2/Conv/weights*(
_output_shapes
:*
shared_name *
dtype0
Ŋ
*generator/G_MODEL/C/r2/Conv/weights/AssignAssign#generator/G_MODEL/C/r2/Conv/weights@generator/G_MODEL/C/r2/Conv/weights/Initializer/truncated_normal*
T0*6
_class,
*(loc:@generator/G_MODEL/C/r2/Conv/weights*(
_output_shapes
:*
use_locking(*
validate_shape(
Ä
(generator/G_MODEL/C/r2/Conv/weights/readIdentity#generator/G_MODEL/C/r2/Conv/weights*
T0*(
_output_shapes
:*6
_class,
*(loc:@generator/G_MODEL/C/r2/Conv/weights
z
)generator/G_MODEL/C/r2/Conv/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
Ė
"generator/G_MODEL/C/r2/Conv/Conv2DConv2D(generator/G_MODEL/C/r1/2/batchnorm/add_1(generator/G_MODEL/C/r2/Conv/weights/read*
strides
*
T0*
paddingVALID*
	dilations
*
explicit_paddings
 *9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
data_formatNHWC*
use_cudnn_on_gpu(
Ā
7generator/G_MODEL/C/r2/LayerNorm/beta/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *
dtype0*8
_class.
,*loc:@generator/G_MODEL/C/r2/LayerNorm/beta
Í
%generator/G_MODEL/C/r2/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shared_name *
shape:*8
_class.
,*loc:@generator/G_MODEL/C/r2/LayerNorm/beta

,generator/G_MODEL/C/r2/LayerNorm/beta/AssignAssign%generator/G_MODEL/C/r2/LayerNorm/beta7generator/G_MODEL/C/r2/LayerNorm/beta/Initializer/zeros*
use_locking(*8
_class.
,*loc:@generator/G_MODEL/C/r2/LayerNorm/beta*
_output_shapes	
:*
validate_shape(*
T0
―
*generator/G_MODEL/C/r2/LayerNorm/beta/readIdentity%generator/G_MODEL/C/r2/LayerNorm/beta*
_output_shapes	
:*
T0*8
_class.
,*loc:@generator/G_MODEL/C/r2/LayerNorm/beta
Á
7generator/G_MODEL/C/r2/LayerNorm/gamma/Initializer/onesConst*
dtype0*9
_class/
-+loc:@generator/G_MODEL/C/r2/LayerNorm/gamma*
valueB*  ?*
_output_shapes	
:
Ï
&generator/G_MODEL/C/r2/LayerNorm/gamma
VariableV2*9
_class/
-+loc:@generator/G_MODEL/C/r2/LayerNorm/gamma*
shape:*
dtype0*
	container *
shared_name *
_output_shapes	
:
Ē
-generator/G_MODEL/C/r2/LayerNorm/gamma/AssignAssign&generator/G_MODEL/C/r2/LayerNorm/gamma7generator/G_MODEL/C/r2/LayerNorm/gamma/Initializer/ones*
_output_shapes	
:*9
_class/
-+loc:@generator/G_MODEL/C/r2/LayerNorm/gamma*
T0*
validate_shape(*
use_locking(
Ā
+generator/G_MODEL/C/r2/LayerNorm/gamma/readIdentity&generator/G_MODEL/C/r2/LayerNorm/gamma*9
_class/
-+loc:@generator/G_MODEL/C/r2/LayerNorm/gamma*
T0*
_output_shapes	
:

?generator/G_MODEL/C/r2/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0
č
-generator/G_MODEL/C/r2/LayerNorm/moments/meanMean"generator/G_MODEL/C/r2/Conv/Conv2D?generator/G_MODEL/C/r2/LayerNorm/moments/mean/reduction_indices*&
_output_shapes
:*
T0*
	keep_dims(*

Tidx0
Ĩ
5generator/G_MODEL/C/r2/LayerNorm/moments/StopGradientStopGradient-generator/G_MODEL/C/r2/LayerNorm/moments/mean*&
_output_shapes
:*
T0
î
:generator/G_MODEL/C/r2/LayerNorm/moments/SquaredDifferenceSquaredDifference"generator/G_MODEL/C/r2/Conv/Conv2D5generator/G_MODEL/C/r2/LayerNorm/moments/StopGradient*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ

Cgenerator/G_MODEL/C/r2/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0

1generator/G_MODEL/C/r2/LayerNorm/moments/varianceMean:generator/G_MODEL/C/r2/LayerNorm/moments/SquaredDifferenceCgenerator/G_MODEL/C/r2/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*&
_output_shapes
:*
T0
u
0generator/G_MODEL/C/r2/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ėž+
Í
.generator/G_MODEL/C/r2/LayerNorm/batchnorm/addAddV21generator/G_MODEL/C/r2/LayerNorm/moments/variance0generator/G_MODEL/C/r2/LayerNorm/batchnorm/add/y*
T0*&
_output_shapes
:

0generator/G_MODEL/C/r2/LayerNorm/batchnorm/RsqrtRsqrt.generator/G_MODEL/C/r2/LayerNorm/batchnorm/add*
T0*&
_output_shapes
:
Æ
.generator/G_MODEL/C/r2/LayerNorm/batchnorm/mulMul0generator/G_MODEL/C/r2/LayerNorm/batchnorm/Rsqrt+generator/G_MODEL/C/r2/LayerNorm/gamma/read*'
_output_shapes
:*
T0
Ï
0generator/G_MODEL/C/r2/LayerNorm/batchnorm/mul_1Mul"generator/G_MODEL/C/r2/Conv/Conv2D.generator/G_MODEL/C/r2/LayerNorm/batchnorm/mul*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
Č
0generator/G_MODEL/C/r2/LayerNorm/batchnorm/mul_2Mul-generator/G_MODEL/C/r2/LayerNorm/moments/mean.generator/G_MODEL/C/r2/LayerNorm/batchnorm/mul*'
_output_shapes
:*
T0
Å
.generator/G_MODEL/C/r2/LayerNorm/batchnorm/subSub*generator/G_MODEL/C/r2/LayerNorm/beta/read0generator/G_MODEL/C/r2/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:
ß
0generator/G_MODEL/C/r2/LayerNorm/batchnorm/add_1AddV20generator/G_MODEL/C/r2/LayerNorm/batchnorm/mul_1.generator/G_MODEL/C/r2/LayerNorm/batchnorm/sub*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
ģ
 generator/G_MODEL/C/r2/LeakyRelu	LeakyRelu0generator/G_MODEL/C/r2/LayerNorm/batchnorm/add_1*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
alpha%ÍĖL>

)generator/G_MODEL/C/r2/MirrorPad/paddingsConst*9
value0B."                             *
_output_shapes

:*
dtype0
â
 generator/G_MODEL/C/r2/MirrorPad	MirrorPad generator/G_MODEL/C/r2/LeakyRelu)generator/G_MODEL/C/r2/MirrorPad/paddings*
T0*
mode	REFLECT*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
	Tpaddings0
Į
>generator/G_MODEL/C/r2/r2/w/Initializer/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0*.
_class$
" loc:@generator/G_MODEL/C/r2/r2/w
ē
=generator/G_MODEL/C/r2/r2/w/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@generator/G_MODEL/C/r2/r2/w
ī
?generator/G_MODEL/C/r2/r2/w/Initializer/truncated_normal/stddevConst*
dtype0*.
_class$
" loc:@generator/G_MODEL/C/r2/r2/w*
_output_shapes
: *
valueB
 *Â<
Ģ
Hgenerator/G_MODEL/C/r2/r2/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>generator/G_MODEL/C/r2/r2/w/Initializer/truncated_normal/shape*.
_class$
" loc:@generator/G_MODEL/C/r2/r2/w*'
_output_shapes
:*

seed *
seed2 *
dtype0*
T0
°
<generator/G_MODEL/C/r2/r2/w/Initializer/truncated_normal/mulMulHgenerator/G_MODEL/C/r2/r2/w/Initializer/truncated_normal/TruncatedNormal?generator/G_MODEL/C/r2/r2/w/Initializer/truncated_normal/stddev*.
_class$
" loc:@generator/G_MODEL/C/r2/r2/w*'
_output_shapes
:*
T0

8generator/G_MODEL/C/r2/r2/w/Initializer/truncated_normalAdd<generator/G_MODEL/C/r2/r2/w/Initializer/truncated_normal/mul=generator/G_MODEL/C/r2/r2/w/Initializer/truncated_normal/mean*'
_output_shapes
:*
T0*.
_class$
" loc:@generator/G_MODEL/C/r2/r2/w
Ņ
generator/G_MODEL/C/r2/r2/w
VariableV2*
shared_name *
	container *
dtype0*
shape:*.
_class$
" loc:@generator/G_MODEL/C/r2/r2/w*'
_output_shapes
:

"generator/G_MODEL/C/r2/r2/w/AssignAssigngenerator/G_MODEL/C/r2/r2/w8generator/G_MODEL/C/r2/r2/w/Initializer/truncated_normal*
T0*.
_class$
" loc:@generator/G_MODEL/C/r2/r2/w*'
_output_shapes
:*
validate_shape(*
use_locking(
Ŧ
 generator/G_MODEL/C/r2/r2/w/readIdentitygenerator/G_MODEL/C/r2/r2/w*.
_class$
" loc:@generator/G_MODEL/C/r2/r2/w*'
_output_shapes
:*
T0
{
"generator/G_MODEL/C/r2/r2/r2/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
{
*generator/G_MODEL/C/r2/r2/r2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      

generator/G_MODEL/C/r2/r2/r2DepthwiseConv2dNative generator/G_MODEL/C/r2/MirrorPad generator/G_MODEL/C/r2/r2/w/read*
T0*
paddingVALID*
	dilations
*
data_formatNHWC*
strides
*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
ē
0generator/G_MODEL/C/r2/r2/bias/Initializer/ConstConst*
valueB*    *
_output_shapes	
:*1
_class'
%#loc:@generator/G_MODEL/C/r2/r2/bias*
dtype0
ŋ
generator/G_MODEL/C/r2/r2/bias
VariableV2*
shape:*
	container *
_output_shapes	
:*
shared_name *
dtype0*1
_class'
%#loc:@generator/G_MODEL/C/r2/r2/bias

%generator/G_MODEL/C/r2/r2/bias/AssignAssigngenerator/G_MODEL/C/r2/r2/bias0generator/G_MODEL/C/r2/r2/bias/Initializer/Const*1
_class'
%#loc:@generator/G_MODEL/C/r2/r2/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
Ļ
#generator/G_MODEL/C/r2/r2/bias/readIdentitygenerator/G_MODEL/C/r2/r2/bias*1
_class'
%#loc:@generator/G_MODEL/C/r2/r2/bias*
_output_shapes	
:*
T0
Ę
!generator/G_MODEL/C/r2/r2/BiasAddBiasAddgenerator/G_MODEL/C/r2/r2/r2#generator/G_MODEL/C/r2/r2/bias/read*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0*
data_formatNHWC
°
/generator/G_MODEL/C/r2/1/beta/Initializer/zerosConst*
dtype0*0
_class&
$"loc:@generator/G_MODEL/C/r2/1/beta*
_output_shapes	
:*
valueB*    
―
generator/G_MODEL/C/r2/1/beta
VariableV2*
	container *
shared_name *
shape:*
dtype0*0
_class&
$"loc:@generator/G_MODEL/C/r2/1/beta*
_output_shapes	
:
ĸ
$generator/G_MODEL/C/r2/1/beta/AssignAssigngenerator/G_MODEL/C/r2/1/beta/generator/G_MODEL/C/r2/1/beta/Initializer/zeros*
use_locking(*
_output_shapes	
:*
T0*0
_class&
$"loc:@generator/G_MODEL/C/r2/1/beta*
validate_shape(
Ĩ
"generator/G_MODEL/C/r2/1/beta/readIdentitygenerator/G_MODEL/C/r2/1/beta*0
_class&
$"loc:@generator/G_MODEL/C/r2/1/beta*
_output_shapes	
:*
T0
ą
/generator/G_MODEL/C/r2/1/gamma/Initializer/onesConst*
dtype0*1
_class'
%#loc:@generator/G_MODEL/C/r2/1/gamma*
valueB*  ?*
_output_shapes	
:
ŋ
generator/G_MODEL/C/r2/1/gamma
VariableV2*
shared_name *1
_class'
%#loc:@generator/G_MODEL/C/r2/1/gamma*
	container *
dtype0*
_output_shapes	
:*
shape:

%generator/G_MODEL/C/r2/1/gamma/AssignAssigngenerator/G_MODEL/C/r2/1/gamma/generator/G_MODEL/C/r2/1/gamma/Initializer/ones*
T0*1
_class'
%#loc:@generator/G_MODEL/C/r2/1/gamma*
use_locking(*
validate_shape(*
_output_shapes	
:
Ļ
#generator/G_MODEL/C/r2/1/gamma/readIdentitygenerator/G_MODEL/C/r2/1/gamma*1
_class'
%#loc:@generator/G_MODEL/C/r2/1/gamma*
_output_shapes	
:*
T0

7generator/G_MODEL/C/r2/1/moments/mean/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0
Ũ
%generator/G_MODEL/C/r2/1/moments/meanMean!generator/G_MODEL/C/r2/r2/BiasAdd7generator/G_MODEL/C/r2/1/moments/mean/reduction_indices*&
_output_shapes
:*
	keep_dims(*

Tidx0*
T0

-generator/G_MODEL/C/r2/1/moments/StopGradientStopGradient%generator/G_MODEL/C/r2/1/moments/mean*&
_output_shapes
:*
T0
Ý
2generator/G_MODEL/C/r2/1/moments/SquaredDifferenceSquaredDifference!generator/G_MODEL/C/r2/r2/BiasAdd-generator/G_MODEL/C/r2/1/moments/StopGradient*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ

;generator/G_MODEL/C/r2/1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
ð
)generator/G_MODEL/C/r2/1/moments/varianceMean2generator/G_MODEL/C/r2/1/moments/SquaredDifference;generator/G_MODEL/C/r2/1/moments/variance/reduction_indices*
	keep_dims(*&
_output_shapes
:*

Tidx0*
T0
m
(generator/G_MODEL/C/r2/1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ėž+
ĩ
&generator/G_MODEL/C/r2/1/batchnorm/addAddV2)generator/G_MODEL/C/r2/1/moments/variance(generator/G_MODEL/C/r2/1/batchnorm/add/y*&
_output_shapes
:*
T0

(generator/G_MODEL/C/r2/1/batchnorm/RsqrtRsqrt&generator/G_MODEL/C/r2/1/batchnorm/add*&
_output_shapes
:*
T0
Ū
&generator/G_MODEL/C/r2/1/batchnorm/mulMul(generator/G_MODEL/C/r2/1/batchnorm/Rsqrt#generator/G_MODEL/C/r2/1/gamma/read*'
_output_shapes
:*
T0
ū
(generator/G_MODEL/C/r2/1/batchnorm/mul_1Mul!generator/G_MODEL/C/r2/r2/BiasAdd&generator/G_MODEL/C/r2/1/batchnorm/mul*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
°
(generator/G_MODEL/C/r2/1/batchnorm/mul_2Mul%generator/G_MODEL/C/r2/1/moments/mean&generator/G_MODEL/C/r2/1/batchnorm/mul*
T0*'
_output_shapes
:
­
&generator/G_MODEL/C/r2/1/batchnorm/subSub"generator/G_MODEL/C/r2/1/beta/read(generator/G_MODEL/C/r2/1/batchnorm/mul_2*'
_output_shapes
:*
T0
Į
(generator/G_MODEL/C/r2/1/batchnorm/add_1AddV2(generator/G_MODEL/C/r2/1/batchnorm/mul_1&generator/G_MODEL/C/r2/1/batchnorm/sub*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
­
"generator/G_MODEL/C/r2/LeakyRelu_1	LeakyRelu(generator/G_MODEL/C/r2/1/batchnorm/add_1*
alpha%ÍĖL>*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Û
Hgenerator/G_MODEL/C/r2/Conv_1/weights/Initializer/truncated_normal/shapeConst*
dtype0*8
_class.
,*loc:@generator/G_MODEL/C/r2/Conv_1/weights*%
valueB"            *
_output_shapes
:
Æ
Ggenerator/G_MODEL/C/r2/Conv_1/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *8
_class.
,*loc:@generator/G_MODEL/C/r2/Conv_1/weights*
_output_shapes
: *
dtype0
Č
Igenerator/G_MODEL/C/r2/Conv_1/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *8
_class.
,*loc:@generator/G_MODEL/C/r2/Conv_1/weights*
valueB
 *Eņ=
Â
Rgenerator/G_MODEL/C/r2/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHgenerator/G_MODEL/C/r2/Conv_1/weights/Initializer/truncated_normal/shape*
T0*
seed2 *

seed *
dtype0*(
_output_shapes
:*8
_class.
,*loc:@generator/G_MODEL/C/r2/Conv_1/weights
Ų
Fgenerator/G_MODEL/C/r2/Conv_1/weights/Initializer/truncated_normal/mulMulRgenerator/G_MODEL/C/r2/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalIgenerator/G_MODEL/C/r2/Conv_1/weights/Initializer/truncated_normal/stddev*8
_class.
,*loc:@generator/G_MODEL/C/r2/Conv_1/weights*
T0*(
_output_shapes
:
Į
Bgenerator/G_MODEL/C/r2/Conv_1/weights/Initializer/truncated_normalAddFgenerator/G_MODEL/C/r2/Conv_1/weights/Initializer/truncated_normal/mulGgenerator/G_MODEL/C/r2/Conv_1/weights/Initializer/truncated_normal/mean*8
_class.
,*loc:@generator/G_MODEL/C/r2/Conv_1/weights*(
_output_shapes
:*
T0
į
%generator/G_MODEL/C/r2/Conv_1/weights
VariableV2*
shape:*
shared_name *
	container *
dtype0*(
_output_shapes
:*8
_class.
,*loc:@generator/G_MODEL/C/r2/Conv_1/weights
·
,generator/G_MODEL/C/r2/Conv_1/weights/AssignAssign%generator/G_MODEL/C/r2/Conv_1/weightsBgenerator/G_MODEL/C/r2/Conv_1/weights/Initializer/truncated_normal*
use_locking(*
validate_shape(*8
_class.
,*loc:@generator/G_MODEL/C/r2/Conv_1/weights*(
_output_shapes
:*
T0
Ę
*generator/G_MODEL/C/r2/Conv_1/weights/readIdentity%generator/G_MODEL/C/r2/Conv_1/weights*8
_class.
,*loc:@generator/G_MODEL/C/r2/Conv_1/weights*(
_output_shapes
:*
T0
|
+generator/G_MODEL/C/r2/Conv_1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
Ę
$generator/G_MODEL/C/r2/Conv_1/Conv2DConv2D"generator/G_MODEL/C/r2/LeakyRelu_1*generator/G_MODEL/C/r2/Conv_1/weights/read*
data_formatNHWC*
	dilations
*
use_cudnn_on_gpu(*
T0*
strides
*
paddingVALID*
explicit_paddings
 *9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
°
/generator/G_MODEL/C/r2/2/beta/Initializer/zerosConst*
valueB*    *
_output_shapes	
:*
dtype0*0
_class&
$"loc:@generator/G_MODEL/C/r2/2/beta
―
generator/G_MODEL/C/r2/2/beta
VariableV2*
shape:*
_output_shapes	
:*0
_class&
$"loc:@generator/G_MODEL/C/r2/2/beta*
dtype0*
	container *
shared_name 
ĸ
$generator/G_MODEL/C/r2/2/beta/AssignAssigngenerator/G_MODEL/C/r2/2/beta/generator/G_MODEL/C/r2/2/beta/Initializer/zeros*
T0*0
_class&
$"loc:@generator/G_MODEL/C/r2/2/beta*
_output_shapes	
:*
validate_shape(*
use_locking(
Ĩ
"generator/G_MODEL/C/r2/2/beta/readIdentitygenerator/G_MODEL/C/r2/2/beta*
_output_shapes	
:*0
_class&
$"loc:@generator/G_MODEL/C/r2/2/beta*
T0
ą
/generator/G_MODEL/C/r2/2/gamma/Initializer/onesConst*
valueB*  ?*
_output_shapes	
:*
dtype0*1
_class'
%#loc:@generator/G_MODEL/C/r2/2/gamma
ŋ
generator/G_MODEL/C/r2/2/gamma
VariableV2*
shape:*
_output_shapes	
:*
shared_name *1
_class'
%#loc:@generator/G_MODEL/C/r2/2/gamma*
	container *
dtype0

%generator/G_MODEL/C/r2/2/gamma/AssignAssigngenerator/G_MODEL/C/r2/2/gamma/generator/G_MODEL/C/r2/2/gamma/Initializer/ones*
use_locking(*1
_class'
%#loc:@generator/G_MODEL/C/r2/2/gamma*
T0*
_output_shapes	
:*
validate_shape(
Ļ
#generator/G_MODEL/C/r2/2/gamma/readIdentitygenerator/G_MODEL/C/r2/2/gamma*1
_class'
%#loc:@generator/G_MODEL/C/r2/2/gamma*
_output_shapes	
:*
T0

7generator/G_MODEL/C/r2/2/moments/mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:
Ú
%generator/G_MODEL/C/r2/2/moments/meanMean$generator/G_MODEL/C/r2/Conv_1/Conv2D7generator/G_MODEL/C/r2/2/moments/mean/reduction_indices*&
_output_shapes
:*
	keep_dims(*
T0*

Tidx0

-generator/G_MODEL/C/r2/2/moments/StopGradientStopGradient%generator/G_MODEL/C/r2/2/moments/mean*&
_output_shapes
:*
T0
ā
2generator/G_MODEL/C/r2/2/moments/SquaredDifferenceSquaredDifference$generator/G_MODEL/C/r2/Conv_1/Conv2D-generator/G_MODEL/C/r2/2/moments/StopGradient*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ

;generator/G_MODEL/C/r2/2/moments/variance/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:
ð
)generator/G_MODEL/C/r2/2/moments/varianceMean2generator/G_MODEL/C/r2/2/moments/SquaredDifference;generator/G_MODEL/C/r2/2/moments/variance/reduction_indices*&
_output_shapes
:*
	keep_dims(*

Tidx0*
T0
m
(generator/G_MODEL/C/r2/2/batchnorm/add/yConst*
dtype0*
valueB
 *Ėž+*
_output_shapes
: 
ĩ
&generator/G_MODEL/C/r2/2/batchnorm/addAddV2)generator/G_MODEL/C/r2/2/moments/variance(generator/G_MODEL/C/r2/2/batchnorm/add/y*
T0*&
_output_shapes
:

(generator/G_MODEL/C/r2/2/batchnorm/RsqrtRsqrt&generator/G_MODEL/C/r2/2/batchnorm/add*&
_output_shapes
:*
T0
Ū
&generator/G_MODEL/C/r2/2/batchnorm/mulMul(generator/G_MODEL/C/r2/2/batchnorm/Rsqrt#generator/G_MODEL/C/r2/2/gamma/read*'
_output_shapes
:*
T0
Á
(generator/G_MODEL/C/r2/2/batchnorm/mul_1Mul$generator/G_MODEL/C/r2/Conv_1/Conv2D&generator/G_MODEL/C/r2/2/batchnorm/mul*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
°
(generator/G_MODEL/C/r2/2/batchnorm/mul_2Mul%generator/G_MODEL/C/r2/2/moments/mean&generator/G_MODEL/C/r2/2/batchnorm/mul*'
_output_shapes
:*
T0
­
&generator/G_MODEL/C/r2/2/batchnorm/subSub"generator/G_MODEL/C/r2/2/beta/read(generator/G_MODEL/C/r2/2/batchnorm/mul_2*
T0*'
_output_shapes
:
Į
(generator/G_MODEL/C/r2/2/batchnorm/add_1AddV2(generator/G_MODEL/C/r2/2/batchnorm/mul_1&generator/G_MODEL/C/r2/2/batchnorm/sub*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
ŧ
generator/G_MODEL/C/r2/addAddV2(generator/G_MODEL/C/r1/2/batchnorm/add_1(generator/G_MODEL/C/r2/2/batchnorm/add_1*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
Ũ
Fgenerator/G_MODEL/C/r3/Conv/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:*6
_class,
*(loc:@generator/G_MODEL/C/r3/Conv/weights
Â
Egenerator/G_MODEL/C/r3/Conv/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *6
_class,
*(loc:@generator/G_MODEL/C/r3/Conv/weights*
valueB
 *    
Ä
Ggenerator/G_MODEL/C/r3/Conv/weights/Initializer/truncated_normal/stddevConst*
valueB
 *ÐdÎ=*6
_class,
*(loc:@generator/G_MODEL/C/r3/Conv/weights*
dtype0*
_output_shapes
: 
ž
Pgenerator/G_MODEL/C/r3/Conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalFgenerator/G_MODEL/C/r3/Conv/weights/Initializer/truncated_normal/shape*6
_class,
*(loc:@generator/G_MODEL/C/r3/Conv/weights*
T0*

seed *
seed2 *(
_output_shapes
:*
dtype0
Ņ
Dgenerator/G_MODEL/C/r3/Conv/weights/Initializer/truncated_normal/mulMulPgenerator/G_MODEL/C/r3/Conv/weights/Initializer/truncated_normal/TruncatedNormalGgenerator/G_MODEL/C/r3/Conv/weights/Initializer/truncated_normal/stddev*6
_class,
*(loc:@generator/G_MODEL/C/r3/Conv/weights*
T0*(
_output_shapes
:
ŋ
@generator/G_MODEL/C/r3/Conv/weights/Initializer/truncated_normalAddDgenerator/G_MODEL/C/r3/Conv/weights/Initializer/truncated_normal/mulEgenerator/G_MODEL/C/r3/Conv/weights/Initializer/truncated_normal/mean*6
_class,
*(loc:@generator/G_MODEL/C/r3/Conv/weights*(
_output_shapes
:*
T0
ã
#generator/G_MODEL/C/r3/Conv/weights
VariableV2*(
_output_shapes
:*
shared_name *
dtype0*
	container *6
_class,
*(loc:@generator/G_MODEL/C/r3/Conv/weights*
shape:
Ŋ
*generator/G_MODEL/C/r3/Conv/weights/AssignAssign#generator/G_MODEL/C/r3/Conv/weights@generator/G_MODEL/C/r3/Conv/weights/Initializer/truncated_normal*(
_output_shapes
:*
use_locking(*6
_class,
*(loc:@generator/G_MODEL/C/r3/Conv/weights*
T0*
validate_shape(
Ä
(generator/G_MODEL/C/r3/Conv/weights/readIdentity#generator/G_MODEL/C/r3/Conv/weights*(
_output_shapes
:*
T0*6
_class,
*(loc:@generator/G_MODEL/C/r3/Conv/weights
z
)generator/G_MODEL/C/r3/Conv/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
ū
"generator/G_MODEL/C/r3/Conv/Conv2DConv2Dgenerator/G_MODEL/C/r2/add(generator/G_MODEL/C/r3/Conv/weights/read*
use_cudnn_on_gpu(*
	dilations
*
paddingVALID*
strides
*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
explicit_paddings
 *
data_formatNHWC
Ā
7generator/G_MODEL/C/r3/LayerNorm/beta/Initializer/zerosConst*8
_class.
,*loc:@generator/G_MODEL/C/r3/LayerNorm/beta*
valueB*    *
_output_shapes	
:*
dtype0
Í
%generator/G_MODEL/C/r3/LayerNorm/beta
VariableV2*
_output_shapes	
:*
shared_name *
dtype0*
	container *8
_class.
,*loc:@generator/G_MODEL/C/r3/LayerNorm/beta*
shape:

,generator/G_MODEL/C/r3/LayerNorm/beta/AssignAssign%generator/G_MODEL/C/r3/LayerNorm/beta7generator/G_MODEL/C/r3/LayerNorm/beta/Initializer/zeros*
_output_shapes	
:*
use_locking(*8
_class.
,*loc:@generator/G_MODEL/C/r3/LayerNorm/beta*
validate_shape(*
T0
―
*generator/G_MODEL/C/r3/LayerNorm/beta/readIdentity%generator/G_MODEL/C/r3/LayerNorm/beta*
T0*
_output_shapes	
:*8
_class.
,*loc:@generator/G_MODEL/C/r3/LayerNorm/beta
Á
7generator/G_MODEL/C/r3/LayerNorm/gamma/Initializer/onesConst*
_output_shapes	
:*
dtype0*9
_class/
-+loc:@generator/G_MODEL/C/r3/LayerNorm/gamma*
valueB*  ?
Ï
&generator/G_MODEL/C/r3/LayerNorm/gamma
VariableV2*
shape:*9
_class/
-+loc:@generator/G_MODEL/C/r3/LayerNorm/gamma*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ē
-generator/G_MODEL/C/r3/LayerNorm/gamma/AssignAssign&generator/G_MODEL/C/r3/LayerNorm/gamma7generator/G_MODEL/C/r3/LayerNorm/gamma/Initializer/ones*9
_class/
-+loc:@generator/G_MODEL/C/r3/LayerNorm/gamma*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
Ā
+generator/G_MODEL/C/r3/LayerNorm/gamma/readIdentity&generator/G_MODEL/C/r3/LayerNorm/gamma*9
_class/
-+loc:@generator/G_MODEL/C/r3/LayerNorm/gamma*
T0*
_output_shapes	
:

?generator/G_MODEL/C/r3/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0
č
-generator/G_MODEL/C/r3/LayerNorm/moments/meanMean"generator/G_MODEL/C/r3/Conv/Conv2D?generator/G_MODEL/C/r3/LayerNorm/moments/mean/reduction_indices*
T0*
	keep_dims(*

Tidx0*&
_output_shapes
:
Ĩ
5generator/G_MODEL/C/r3/LayerNorm/moments/StopGradientStopGradient-generator/G_MODEL/C/r3/LayerNorm/moments/mean*
T0*&
_output_shapes
:
î
:generator/G_MODEL/C/r3/LayerNorm/moments/SquaredDifferenceSquaredDifference"generator/G_MODEL/C/r3/Conv/Conv2D5generator/G_MODEL/C/r3/LayerNorm/moments/StopGradient*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ

Cgenerator/G_MODEL/C/r3/LayerNorm/moments/variance/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0

1generator/G_MODEL/C/r3/LayerNorm/moments/varianceMean:generator/G_MODEL/C/r3/LayerNorm/moments/SquaredDifferenceCgenerator/G_MODEL/C/r3/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*&
_output_shapes
:*

Tidx0*
T0
u
0generator/G_MODEL/C/r3/LayerNorm/batchnorm/add/yConst*
dtype0*
valueB
 *Ėž+*
_output_shapes
: 
Í
.generator/G_MODEL/C/r3/LayerNorm/batchnorm/addAddV21generator/G_MODEL/C/r3/LayerNorm/moments/variance0generator/G_MODEL/C/r3/LayerNorm/batchnorm/add/y*&
_output_shapes
:*
T0

0generator/G_MODEL/C/r3/LayerNorm/batchnorm/RsqrtRsqrt.generator/G_MODEL/C/r3/LayerNorm/batchnorm/add*&
_output_shapes
:*
T0
Æ
.generator/G_MODEL/C/r3/LayerNorm/batchnorm/mulMul0generator/G_MODEL/C/r3/LayerNorm/batchnorm/Rsqrt+generator/G_MODEL/C/r3/LayerNorm/gamma/read*'
_output_shapes
:*
T0
Ï
0generator/G_MODEL/C/r3/LayerNorm/batchnorm/mul_1Mul"generator/G_MODEL/C/r3/Conv/Conv2D.generator/G_MODEL/C/r3/LayerNorm/batchnorm/mul*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Č
0generator/G_MODEL/C/r3/LayerNorm/batchnorm/mul_2Mul-generator/G_MODEL/C/r3/LayerNorm/moments/mean.generator/G_MODEL/C/r3/LayerNorm/batchnorm/mul*'
_output_shapes
:*
T0
Å
.generator/G_MODEL/C/r3/LayerNorm/batchnorm/subSub*generator/G_MODEL/C/r3/LayerNorm/beta/read0generator/G_MODEL/C/r3/LayerNorm/batchnorm/mul_2*'
_output_shapes
:*
T0
ß
0generator/G_MODEL/C/r3/LayerNorm/batchnorm/add_1AddV20generator/G_MODEL/C/r3/LayerNorm/batchnorm/mul_1.generator/G_MODEL/C/r3/LayerNorm/batchnorm/sub*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
ģ
 generator/G_MODEL/C/r3/LeakyRelu	LeakyRelu0generator/G_MODEL/C/r3/LayerNorm/batchnorm/add_1*
alpha%ÍĖL>*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0

)generator/G_MODEL/C/r3/MirrorPad/paddingsConst*
dtype0*9
value0B."                             *
_output_shapes

:
â
 generator/G_MODEL/C/r3/MirrorPad	MirrorPad generator/G_MODEL/C/r3/LeakyRelu)generator/G_MODEL/C/r3/MirrorPad/paddings*
mode	REFLECT*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0*
	Tpaddings0
Į
>generator/G_MODEL/C/r3/r3/w/Initializer/truncated_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:*.
_class$
" loc:@generator/G_MODEL/C/r3/r3/w
ē
=generator/G_MODEL/C/r3/r3/w/Initializer/truncated_normal/meanConst*
_output_shapes
: *.
_class$
" loc:@generator/G_MODEL/C/r3/r3/w*
dtype0*
valueB
 *    
ī
?generator/G_MODEL/C/r3/r3/w/Initializer/truncated_normal/stddevConst*
_output_shapes
: *.
_class$
" loc:@generator/G_MODEL/C/r3/r3/w*
dtype0*
valueB
 *Â<
Ģ
Hgenerator/G_MODEL/C/r3/r3/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>generator/G_MODEL/C/r3/r3/w/Initializer/truncated_normal/shape*

seed *
dtype0*
T0*.
_class$
" loc:@generator/G_MODEL/C/r3/r3/w*
seed2 *'
_output_shapes
:
°
<generator/G_MODEL/C/r3/r3/w/Initializer/truncated_normal/mulMulHgenerator/G_MODEL/C/r3/r3/w/Initializer/truncated_normal/TruncatedNormal?generator/G_MODEL/C/r3/r3/w/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@generator/G_MODEL/C/r3/r3/w*'
_output_shapes
:

8generator/G_MODEL/C/r3/r3/w/Initializer/truncated_normalAdd<generator/G_MODEL/C/r3/r3/w/Initializer/truncated_normal/mul=generator/G_MODEL/C/r3/r3/w/Initializer/truncated_normal/mean*
T0*'
_output_shapes
:*.
_class$
" loc:@generator/G_MODEL/C/r3/r3/w
Ņ
generator/G_MODEL/C/r3/r3/w
VariableV2*
shared_name *'
_output_shapes
:*
dtype0*
shape:*.
_class$
" loc:@generator/G_MODEL/C/r3/r3/w*
	container 

"generator/G_MODEL/C/r3/r3/w/AssignAssigngenerator/G_MODEL/C/r3/r3/w8generator/G_MODEL/C/r3/r3/w/Initializer/truncated_normal*
use_locking(*.
_class$
" loc:@generator/G_MODEL/C/r3/r3/w*
validate_shape(*'
_output_shapes
:*
T0
Ŧ
 generator/G_MODEL/C/r3/r3/w/readIdentitygenerator/G_MODEL/C/r3/r3/w*'
_output_shapes
:*.
_class$
" loc:@generator/G_MODEL/C/r3/r3/w*
T0
{
"generator/G_MODEL/C/r3/r3/r3/ShapeConst*
_output_shapes
:*%
valueB"            *
dtype0
{
*generator/G_MODEL/C/r3/r3/r3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      

generator/G_MODEL/C/r3/r3/r3DepthwiseConv2dNative generator/G_MODEL/C/r3/MirrorPad generator/G_MODEL/C/r3/r3/w/read*
T0*
paddingVALID*
strides
*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
	dilations
*
data_formatNHWC
ē
0generator/G_MODEL/C/r3/r3/bias/Initializer/ConstConst*
valueB*    *1
_class'
%#loc:@generator/G_MODEL/C/r3/r3/bias*
dtype0*
_output_shapes	
:
ŋ
generator/G_MODEL/C/r3/r3/bias
VariableV2*
shape:*
shared_name *
	container *1
_class'
%#loc:@generator/G_MODEL/C/r3/r3/bias*
_output_shapes	
:*
dtype0

%generator/G_MODEL/C/r3/r3/bias/AssignAssigngenerator/G_MODEL/C/r3/r3/bias0generator/G_MODEL/C/r3/r3/bias/Initializer/Const*1
_class'
%#loc:@generator/G_MODEL/C/r3/r3/bias*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(
Ļ
#generator/G_MODEL/C/r3/r3/bias/readIdentitygenerator/G_MODEL/C/r3/r3/bias*1
_class'
%#loc:@generator/G_MODEL/C/r3/r3/bias*
T0*
_output_shapes	
:
Ę
!generator/G_MODEL/C/r3/r3/BiasAddBiasAddgenerator/G_MODEL/C/r3/r3/r3#generator/G_MODEL/C/r3/r3/bias/read*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
data_formatNHWC*
T0
°
/generator/G_MODEL/C/r3/1/beta/Initializer/zerosConst*
_output_shapes	
:*
dtype0*0
_class&
$"loc:@generator/G_MODEL/C/r3/1/beta*
valueB*    
―
generator/G_MODEL/C/r3/1/beta
VariableV2*
	container *
_output_shapes	
:*
shape:*0
_class&
$"loc:@generator/G_MODEL/C/r3/1/beta*
dtype0*
shared_name 
ĸ
$generator/G_MODEL/C/r3/1/beta/AssignAssigngenerator/G_MODEL/C/r3/1/beta/generator/G_MODEL/C/r3/1/beta/Initializer/zeros*
_output_shapes	
:*
T0*0
_class&
$"loc:@generator/G_MODEL/C/r3/1/beta*
use_locking(*
validate_shape(
Ĩ
"generator/G_MODEL/C/r3/1/beta/readIdentitygenerator/G_MODEL/C/r3/1/beta*
T0*
_output_shapes	
:*0
_class&
$"loc:@generator/G_MODEL/C/r3/1/beta
ą
/generator/G_MODEL/C/r3/1/gamma/Initializer/onesConst*
_output_shapes	
:*
dtype0*1
_class'
%#loc:@generator/G_MODEL/C/r3/1/gamma*
valueB*  ?
ŋ
generator/G_MODEL/C/r3/1/gamma
VariableV2*1
_class'
%#loc:@generator/G_MODEL/C/r3/1/gamma*
_output_shapes	
:*
	container *
shape:*
dtype0*
shared_name 

%generator/G_MODEL/C/r3/1/gamma/AssignAssigngenerator/G_MODEL/C/r3/1/gamma/generator/G_MODEL/C/r3/1/gamma/Initializer/ones*1
_class'
%#loc:@generator/G_MODEL/C/r3/1/gamma*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
Ļ
#generator/G_MODEL/C/r3/1/gamma/readIdentitygenerator/G_MODEL/C/r3/1/gamma*
_output_shapes	
:*
T0*1
_class'
%#loc:@generator/G_MODEL/C/r3/1/gamma

7generator/G_MODEL/C/r3/1/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0
Ũ
%generator/G_MODEL/C/r3/1/moments/meanMean!generator/G_MODEL/C/r3/r3/BiasAdd7generator/G_MODEL/C/r3/1/moments/mean/reduction_indices*
T0*

Tidx0*&
_output_shapes
:*
	keep_dims(

-generator/G_MODEL/C/r3/1/moments/StopGradientStopGradient%generator/G_MODEL/C/r3/1/moments/mean*
T0*&
_output_shapes
:
Ý
2generator/G_MODEL/C/r3/1/moments/SquaredDifferenceSquaredDifference!generator/G_MODEL/C/r3/r3/BiasAdd-generator/G_MODEL/C/r3/1/moments/StopGradient*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ

;generator/G_MODEL/C/r3/1/moments/variance/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0
ð
)generator/G_MODEL/C/r3/1/moments/varianceMean2generator/G_MODEL/C/r3/1/moments/SquaredDifference;generator/G_MODEL/C/r3/1/moments/variance/reduction_indices*
T0*
	keep_dims(*

Tidx0*&
_output_shapes
:
m
(generator/G_MODEL/C/r3/1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ėž+
ĩ
&generator/G_MODEL/C/r3/1/batchnorm/addAddV2)generator/G_MODEL/C/r3/1/moments/variance(generator/G_MODEL/C/r3/1/batchnorm/add/y*&
_output_shapes
:*
T0

(generator/G_MODEL/C/r3/1/batchnorm/RsqrtRsqrt&generator/G_MODEL/C/r3/1/batchnorm/add*
T0*&
_output_shapes
:
Ū
&generator/G_MODEL/C/r3/1/batchnorm/mulMul(generator/G_MODEL/C/r3/1/batchnorm/Rsqrt#generator/G_MODEL/C/r3/1/gamma/read*'
_output_shapes
:*
T0
ū
(generator/G_MODEL/C/r3/1/batchnorm/mul_1Mul!generator/G_MODEL/C/r3/r3/BiasAdd&generator/G_MODEL/C/r3/1/batchnorm/mul*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
°
(generator/G_MODEL/C/r3/1/batchnorm/mul_2Mul%generator/G_MODEL/C/r3/1/moments/mean&generator/G_MODEL/C/r3/1/batchnorm/mul*'
_output_shapes
:*
T0
­
&generator/G_MODEL/C/r3/1/batchnorm/subSub"generator/G_MODEL/C/r3/1/beta/read(generator/G_MODEL/C/r3/1/batchnorm/mul_2*'
_output_shapes
:*
T0
Į
(generator/G_MODEL/C/r3/1/batchnorm/add_1AddV2(generator/G_MODEL/C/r3/1/batchnorm/mul_1&generator/G_MODEL/C/r3/1/batchnorm/sub*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
­
"generator/G_MODEL/C/r3/LeakyRelu_1	LeakyRelu(generator/G_MODEL/C/r3/1/batchnorm/add_1*
alpha%ÍĖL>*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Û
Hgenerator/G_MODEL/C/r3/Conv_1/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:*8
_class.
,*loc:@generator/G_MODEL/C/r3/Conv_1/weights
Æ
Ggenerator/G_MODEL/C/r3/Conv_1/weights/Initializer/truncated_normal/meanConst*8
_class.
,*loc:@generator/G_MODEL/C/r3/Conv_1/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
Č
Igenerator/G_MODEL/C/r3/Conv_1/weights/Initializer/truncated_normal/stddevConst*8
_class.
,*loc:@generator/G_MODEL/C/r3/Conv_1/weights*
valueB
 *Eņ=*
dtype0*
_output_shapes
: 
Â
Rgenerator/G_MODEL/C/r3/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHgenerator/G_MODEL/C/r3/Conv_1/weights/Initializer/truncated_normal/shape*
T0*(
_output_shapes
:*8
_class.
,*loc:@generator/G_MODEL/C/r3/Conv_1/weights*
dtype0*
seed2 *

seed 
Ų
Fgenerator/G_MODEL/C/r3/Conv_1/weights/Initializer/truncated_normal/mulMulRgenerator/G_MODEL/C/r3/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalIgenerator/G_MODEL/C/r3/Conv_1/weights/Initializer/truncated_normal/stddev*8
_class.
,*loc:@generator/G_MODEL/C/r3/Conv_1/weights*
T0*(
_output_shapes
:
Į
Bgenerator/G_MODEL/C/r3/Conv_1/weights/Initializer/truncated_normalAddFgenerator/G_MODEL/C/r3/Conv_1/weights/Initializer/truncated_normal/mulGgenerator/G_MODEL/C/r3/Conv_1/weights/Initializer/truncated_normal/mean*
T0*(
_output_shapes
:*8
_class.
,*loc:@generator/G_MODEL/C/r3/Conv_1/weights
į
%generator/G_MODEL/C/r3/Conv_1/weights
VariableV2*8
_class.
,*loc:@generator/G_MODEL/C/r3/Conv_1/weights*
shared_name *
shape:*
	container *
dtype0*(
_output_shapes
:
·
,generator/G_MODEL/C/r3/Conv_1/weights/AssignAssign%generator/G_MODEL/C/r3/Conv_1/weightsBgenerator/G_MODEL/C/r3/Conv_1/weights/Initializer/truncated_normal*
T0*
validate_shape(*(
_output_shapes
:*8
_class.
,*loc:@generator/G_MODEL/C/r3/Conv_1/weights*
use_locking(
Ę
*generator/G_MODEL/C/r3/Conv_1/weights/readIdentity%generator/G_MODEL/C/r3/Conv_1/weights*8
_class.
,*loc:@generator/G_MODEL/C/r3/Conv_1/weights*
T0*(
_output_shapes
:
|
+generator/G_MODEL/C/r3/Conv_1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
Ę
$generator/G_MODEL/C/r3/Conv_1/Conv2DConv2D"generator/G_MODEL/C/r3/LeakyRelu_1*generator/G_MODEL/C/r3/Conv_1/weights/read*
T0*
strides
*
use_cudnn_on_gpu(*
data_formatNHWC*
explicit_paddings
 *
paddingVALID*
	dilations
*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
°
/generator/G_MODEL/C/r3/2/beta/Initializer/zerosConst*
dtype0*0
_class&
$"loc:@generator/G_MODEL/C/r3/2/beta*
valueB*    *
_output_shapes	
:
―
generator/G_MODEL/C/r3/2/beta
VariableV2*
_output_shapes	
:*
dtype0*
shape:*0
_class&
$"loc:@generator/G_MODEL/C/r3/2/beta*
	container *
shared_name 
ĸ
$generator/G_MODEL/C/r3/2/beta/AssignAssigngenerator/G_MODEL/C/r3/2/beta/generator/G_MODEL/C/r3/2/beta/Initializer/zeros*
use_locking(*
_output_shapes	
:*0
_class&
$"loc:@generator/G_MODEL/C/r3/2/beta*
T0*
validate_shape(
Ĩ
"generator/G_MODEL/C/r3/2/beta/readIdentitygenerator/G_MODEL/C/r3/2/beta*
_output_shapes	
:*
T0*0
_class&
$"loc:@generator/G_MODEL/C/r3/2/beta
ą
/generator/G_MODEL/C/r3/2/gamma/Initializer/onesConst*
valueB*  ?*
_output_shapes	
:*
dtype0*1
_class'
%#loc:@generator/G_MODEL/C/r3/2/gamma
ŋ
generator/G_MODEL/C/r3/2/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
	container *
shape:*1
_class'
%#loc:@generator/G_MODEL/C/r3/2/gamma

%generator/G_MODEL/C/r3/2/gamma/AssignAssigngenerator/G_MODEL/C/r3/2/gamma/generator/G_MODEL/C/r3/2/gamma/Initializer/ones*
_output_shapes	
:*1
_class'
%#loc:@generator/G_MODEL/C/r3/2/gamma*
use_locking(*
T0*
validate_shape(
Ļ
#generator/G_MODEL/C/r3/2/gamma/readIdentitygenerator/G_MODEL/C/r3/2/gamma*
T0*
_output_shapes	
:*1
_class'
%#loc:@generator/G_MODEL/C/r3/2/gamma

7generator/G_MODEL/C/r3/2/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0
Ú
%generator/G_MODEL/C/r3/2/moments/meanMean$generator/G_MODEL/C/r3/Conv_1/Conv2D7generator/G_MODEL/C/r3/2/moments/mean/reduction_indices*
T0*
	keep_dims(*&
_output_shapes
:*

Tidx0

-generator/G_MODEL/C/r3/2/moments/StopGradientStopGradient%generator/G_MODEL/C/r3/2/moments/mean*&
_output_shapes
:*
T0
ā
2generator/G_MODEL/C/r3/2/moments/SquaredDifferenceSquaredDifference$generator/G_MODEL/C/r3/Conv_1/Conv2D-generator/G_MODEL/C/r3/2/moments/StopGradient*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ

;generator/G_MODEL/C/r3/2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
ð
)generator/G_MODEL/C/r3/2/moments/varianceMean2generator/G_MODEL/C/r3/2/moments/SquaredDifference;generator/G_MODEL/C/r3/2/moments/variance/reduction_indices*
	keep_dims(*&
_output_shapes
:*
T0*

Tidx0
m
(generator/G_MODEL/C/r3/2/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ėž+
ĩ
&generator/G_MODEL/C/r3/2/batchnorm/addAddV2)generator/G_MODEL/C/r3/2/moments/variance(generator/G_MODEL/C/r3/2/batchnorm/add/y*&
_output_shapes
:*
T0

(generator/G_MODEL/C/r3/2/batchnorm/RsqrtRsqrt&generator/G_MODEL/C/r3/2/batchnorm/add*
T0*&
_output_shapes
:
Ū
&generator/G_MODEL/C/r3/2/batchnorm/mulMul(generator/G_MODEL/C/r3/2/batchnorm/Rsqrt#generator/G_MODEL/C/r3/2/gamma/read*'
_output_shapes
:*
T0
Á
(generator/G_MODEL/C/r3/2/batchnorm/mul_1Mul$generator/G_MODEL/C/r3/Conv_1/Conv2D&generator/G_MODEL/C/r3/2/batchnorm/mul*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
°
(generator/G_MODEL/C/r3/2/batchnorm/mul_2Mul%generator/G_MODEL/C/r3/2/moments/mean&generator/G_MODEL/C/r3/2/batchnorm/mul*
T0*'
_output_shapes
:
­
&generator/G_MODEL/C/r3/2/batchnorm/subSub"generator/G_MODEL/C/r3/2/beta/read(generator/G_MODEL/C/r3/2/batchnorm/mul_2*
T0*'
_output_shapes
:
Į
(generator/G_MODEL/C/r3/2/batchnorm/add_1AddV2(generator/G_MODEL/C/r3/2/batchnorm/mul_1&generator/G_MODEL/C/r3/2/batchnorm/sub*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
­
generator/G_MODEL/C/r3/addAddV2generator/G_MODEL/C/r2/add(generator/G_MODEL/C/r3/2/batchnorm/add_1*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Ũ
Fgenerator/G_MODEL/C/r4/Conv/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            *6
_class,
*(loc:@generator/G_MODEL/C/r4/Conv/weights
Â
Egenerator/G_MODEL/C/r4/Conv/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *6
_class,
*(loc:@generator/G_MODEL/C/r4/Conv/weights*
dtype0
Ä
Ggenerator/G_MODEL/C/r4/Conv/weights/Initializer/truncated_normal/stddevConst*
valueB
 *ÐdÎ=*
dtype0*6
_class,
*(loc:@generator/G_MODEL/C/r4/Conv/weights*
_output_shapes
: 
ž
Pgenerator/G_MODEL/C/r4/Conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalFgenerator/G_MODEL/C/r4/Conv/weights/Initializer/truncated_normal/shape*
T0*

seed *6
_class,
*(loc:@generator/G_MODEL/C/r4/Conv/weights*
seed2 *(
_output_shapes
:*
dtype0
Ņ
Dgenerator/G_MODEL/C/r4/Conv/weights/Initializer/truncated_normal/mulMulPgenerator/G_MODEL/C/r4/Conv/weights/Initializer/truncated_normal/TruncatedNormalGgenerator/G_MODEL/C/r4/Conv/weights/Initializer/truncated_normal/stddev*6
_class,
*(loc:@generator/G_MODEL/C/r4/Conv/weights*
T0*(
_output_shapes
:
ŋ
@generator/G_MODEL/C/r4/Conv/weights/Initializer/truncated_normalAddDgenerator/G_MODEL/C/r4/Conv/weights/Initializer/truncated_normal/mulEgenerator/G_MODEL/C/r4/Conv/weights/Initializer/truncated_normal/mean*6
_class,
*(loc:@generator/G_MODEL/C/r4/Conv/weights*
T0*(
_output_shapes
:
ã
#generator/G_MODEL/C/r4/Conv/weights
VariableV2*(
_output_shapes
:*
shared_name *6
_class,
*(loc:@generator/G_MODEL/C/r4/Conv/weights*
shape:*
	container *
dtype0
Ŋ
*generator/G_MODEL/C/r4/Conv/weights/AssignAssign#generator/G_MODEL/C/r4/Conv/weights@generator/G_MODEL/C/r4/Conv/weights/Initializer/truncated_normal*(
_output_shapes
:*6
_class,
*(loc:@generator/G_MODEL/C/r4/Conv/weights*
use_locking(*
validate_shape(*
T0
Ä
(generator/G_MODEL/C/r4/Conv/weights/readIdentity#generator/G_MODEL/C/r4/Conv/weights*
T0*6
_class,
*(loc:@generator/G_MODEL/C/r4/Conv/weights*(
_output_shapes
:
z
)generator/G_MODEL/C/r4/Conv/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
ū
"generator/G_MODEL/C/r4/Conv/Conv2DConv2Dgenerator/G_MODEL/C/r3/add(generator/G_MODEL/C/r4/Conv/weights/read*
explicit_paddings
 *
	dilations
*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
strides
*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
T0
Ā
7generator/G_MODEL/C/r4/LayerNorm/beta/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *8
_class.
,*loc:@generator/G_MODEL/C/r4/LayerNorm/beta*
dtype0
Í
%generator/G_MODEL/C/r4/LayerNorm/beta
VariableV2*8
_class.
,*loc:@generator/G_MODEL/C/r4/LayerNorm/beta*
_output_shapes	
:*
shared_name *
	container *
dtype0*
shape:

,generator/G_MODEL/C/r4/LayerNorm/beta/AssignAssign%generator/G_MODEL/C/r4/LayerNorm/beta7generator/G_MODEL/C/r4/LayerNorm/beta/Initializer/zeros*
use_locking(*8
_class.
,*loc:@generator/G_MODEL/C/r4/LayerNorm/beta*
_output_shapes	
:*
validate_shape(*
T0
―
*generator/G_MODEL/C/r4/LayerNorm/beta/readIdentity%generator/G_MODEL/C/r4/LayerNorm/beta*
_output_shapes	
:*8
_class.
,*loc:@generator/G_MODEL/C/r4/LayerNorm/beta*
T0
Á
7generator/G_MODEL/C/r4/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*9
_class/
-+loc:@generator/G_MODEL/C/r4/LayerNorm/gamma*
dtype0*
_output_shapes	
:
Ï
&generator/G_MODEL/C/r4/LayerNorm/gamma
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
	container *9
_class/
-+loc:@generator/G_MODEL/C/r4/LayerNorm/gamma*
shared_name 
Ē
-generator/G_MODEL/C/r4/LayerNorm/gamma/AssignAssign&generator/G_MODEL/C/r4/LayerNorm/gamma7generator/G_MODEL/C/r4/LayerNorm/gamma/Initializer/ones*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(*9
_class/
-+loc:@generator/G_MODEL/C/r4/LayerNorm/gamma
Ā
+generator/G_MODEL/C/r4/LayerNorm/gamma/readIdentity&generator/G_MODEL/C/r4/LayerNorm/gamma*
T0*9
_class/
-+loc:@generator/G_MODEL/C/r4/LayerNorm/gamma*
_output_shapes	
:

?generator/G_MODEL/C/r4/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0
č
-generator/G_MODEL/C/r4/LayerNorm/moments/meanMean"generator/G_MODEL/C/r4/Conv/Conv2D?generator/G_MODEL/C/r4/LayerNorm/moments/mean/reduction_indices*

Tidx0*
T0*&
_output_shapes
:*
	keep_dims(
Ĩ
5generator/G_MODEL/C/r4/LayerNorm/moments/StopGradientStopGradient-generator/G_MODEL/C/r4/LayerNorm/moments/mean*&
_output_shapes
:*
T0
î
:generator/G_MODEL/C/r4/LayerNorm/moments/SquaredDifferenceSquaredDifference"generator/G_MODEL/C/r4/Conv/Conv2D5generator/G_MODEL/C/r4/LayerNorm/moments/StopGradient*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0

Cgenerator/G_MODEL/C/r4/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*!
valueB"         *
_output_shapes
:

1generator/G_MODEL/C/r4/LayerNorm/moments/varianceMean:generator/G_MODEL/C/r4/LayerNorm/moments/SquaredDifferenceCgenerator/G_MODEL/C/r4/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*
T0*

Tidx0*&
_output_shapes
:
u
0generator/G_MODEL/C/r4/LayerNorm/batchnorm/add/yConst*
dtype0*
valueB
 *Ėž+*
_output_shapes
: 
Í
.generator/G_MODEL/C/r4/LayerNorm/batchnorm/addAddV21generator/G_MODEL/C/r4/LayerNorm/moments/variance0generator/G_MODEL/C/r4/LayerNorm/batchnorm/add/y*&
_output_shapes
:*
T0

0generator/G_MODEL/C/r4/LayerNorm/batchnorm/RsqrtRsqrt.generator/G_MODEL/C/r4/LayerNorm/batchnorm/add*
T0*&
_output_shapes
:
Æ
.generator/G_MODEL/C/r4/LayerNorm/batchnorm/mulMul0generator/G_MODEL/C/r4/LayerNorm/batchnorm/Rsqrt+generator/G_MODEL/C/r4/LayerNorm/gamma/read*'
_output_shapes
:*
T0
Ï
0generator/G_MODEL/C/r4/LayerNorm/batchnorm/mul_1Mul"generator/G_MODEL/C/r4/Conv/Conv2D.generator/G_MODEL/C/r4/LayerNorm/batchnorm/mul*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
Č
0generator/G_MODEL/C/r4/LayerNorm/batchnorm/mul_2Mul-generator/G_MODEL/C/r4/LayerNorm/moments/mean.generator/G_MODEL/C/r4/LayerNorm/batchnorm/mul*'
_output_shapes
:*
T0
Å
.generator/G_MODEL/C/r4/LayerNorm/batchnorm/subSub*generator/G_MODEL/C/r4/LayerNorm/beta/read0generator/G_MODEL/C/r4/LayerNorm/batchnorm/mul_2*'
_output_shapes
:*
T0
ß
0generator/G_MODEL/C/r4/LayerNorm/batchnorm/add_1AddV20generator/G_MODEL/C/r4/LayerNorm/batchnorm/mul_1.generator/G_MODEL/C/r4/LayerNorm/batchnorm/sub*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
ģ
 generator/G_MODEL/C/r4/LeakyRelu	LeakyRelu0generator/G_MODEL/C/r4/LayerNorm/batchnorm/add_1*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
alpha%ÍĖL>

)generator/G_MODEL/C/r4/MirrorPad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
â
 generator/G_MODEL/C/r4/MirrorPad	MirrorPad generator/G_MODEL/C/r4/LeakyRelu)generator/G_MODEL/C/r4/MirrorPad/paddings*
	Tpaddings0*
mode	REFLECT*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
Į
>generator/G_MODEL/C/r4/r4/w/Initializer/truncated_normal/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0*.
_class$
" loc:@generator/G_MODEL/C/r4/r4/w
ē
=generator/G_MODEL/C/r4/r4/w/Initializer/truncated_normal/meanConst*
dtype0*.
_class$
" loc:@generator/G_MODEL/C/r4/r4/w*
_output_shapes
: *
valueB
 *    
ī
?generator/G_MODEL/C/r4/r4/w/Initializer/truncated_normal/stddevConst*
dtype0*
valueB
 *Â<*
_output_shapes
: *.
_class$
" loc:@generator/G_MODEL/C/r4/r4/w
Ģ
Hgenerator/G_MODEL/C/r4/r4/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>generator/G_MODEL/C/r4/r4/w/Initializer/truncated_normal/shape*'
_output_shapes
:*
T0*
dtype0*.
_class$
" loc:@generator/G_MODEL/C/r4/r4/w*

seed *
seed2 
°
<generator/G_MODEL/C/r4/r4/w/Initializer/truncated_normal/mulMulHgenerator/G_MODEL/C/r4/r4/w/Initializer/truncated_normal/TruncatedNormal?generator/G_MODEL/C/r4/r4/w/Initializer/truncated_normal/stddev*
T0*'
_output_shapes
:*.
_class$
" loc:@generator/G_MODEL/C/r4/r4/w

8generator/G_MODEL/C/r4/r4/w/Initializer/truncated_normalAdd<generator/G_MODEL/C/r4/r4/w/Initializer/truncated_normal/mul=generator/G_MODEL/C/r4/r4/w/Initializer/truncated_normal/mean*
T0*'
_output_shapes
:*.
_class$
" loc:@generator/G_MODEL/C/r4/r4/w
Ņ
generator/G_MODEL/C/r4/r4/w
VariableV2*
	container *'
_output_shapes
:*
dtype0*
shared_name *
shape:*.
_class$
" loc:@generator/G_MODEL/C/r4/r4/w

"generator/G_MODEL/C/r4/r4/w/AssignAssigngenerator/G_MODEL/C/r4/r4/w8generator/G_MODEL/C/r4/r4/w/Initializer/truncated_normal*'
_output_shapes
:*.
_class$
" loc:@generator/G_MODEL/C/r4/r4/w*
use_locking(*
T0*
validate_shape(
Ŧ
 generator/G_MODEL/C/r4/r4/w/readIdentitygenerator/G_MODEL/C/r4/r4/w*
T0*.
_class$
" loc:@generator/G_MODEL/C/r4/r4/w*'
_output_shapes
:
{
"generator/G_MODEL/C/r4/r4/r4/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
{
*generator/G_MODEL/C/r4/r4/r4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

generator/G_MODEL/C/r4/r4/r4DepthwiseConv2dNative generator/G_MODEL/C/r4/MirrorPad generator/G_MODEL/C/r4/r4/w/read*
strides
*
	dilations
*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
paddingVALID*
T0*
data_formatNHWC
ē
0generator/G_MODEL/C/r4/r4/bias/Initializer/ConstConst*
_output_shapes	
:*
valueB*    *
dtype0*1
_class'
%#loc:@generator/G_MODEL/C/r4/r4/bias
ŋ
generator/G_MODEL/C/r4/r4/bias
VariableV2*
	container *
dtype0*1
_class'
%#loc:@generator/G_MODEL/C/r4/r4/bias*
_output_shapes	
:*
shared_name *
shape:

%generator/G_MODEL/C/r4/r4/bias/AssignAssigngenerator/G_MODEL/C/r4/r4/bias0generator/G_MODEL/C/r4/r4/bias/Initializer/Const*1
_class'
%#loc:@generator/G_MODEL/C/r4/r4/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
Ļ
#generator/G_MODEL/C/r4/r4/bias/readIdentitygenerator/G_MODEL/C/r4/r4/bias*
T0*1
_class'
%#loc:@generator/G_MODEL/C/r4/r4/bias*
_output_shapes	
:
Ę
!generator/G_MODEL/C/r4/r4/BiasAddBiasAddgenerator/G_MODEL/C/r4/r4/r4#generator/G_MODEL/C/r4/r4/bias/read*
data_formatNHWC*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
°
/generator/G_MODEL/C/r4/1/beta/Initializer/zerosConst*
_output_shapes	
:*0
_class&
$"loc:@generator/G_MODEL/C/r4/1/beta*
valueB*    *
dtype0
―
generator/G_MODEL/C/r4/1/beta
VariableV2*
shared_name *
	container *
dtype0*
_output_shapes	
:*0
_class&
$"loc:@generator/G_MODEL/C/r4/1/beta*
shape:
ĸ
$generator/G_MODEL/C/r4/1/beta/AssignAssigngenerator/G_MODEL/C/r4/1/beta/generator/G_MODEL/C/r4/1/beta/Initializer/zeros*0
_class&
$"loc:@generator/G_MODEL/C/r4/1/beta*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0
Ĩ
"generator/G_MODEL/C/r4/1/beta/readIdentitygenerator/G_MODEL/C/r4/1/beta*0
_class&
$"loc:@generator/G_MODEL/C/r4/1/beta*
T0*
_output_shapes	
:
ą
/generator/G_MODEL/C/r4/1/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*
valueB*  ?*1
_class'
%#loc:@generator/G_MODEL/C/r4/1/gamma
ŋ
generator/G_MODEL/C/r4/1/gamma
VariableV2*
	container *1
_class'
%#loc:@generator/G_MODEL/C/r4/1/gamma*
shape:*
dtype0*
_output_shapes	
:*
shared_name 

%generator/G_MODEL/C/r4/1/gamma/AssignAssigngenerator/G_MODEL/C/r4/1/gamma/generator/G_MODEL/C/r4/1/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:*1
_class'
%#loc:@generator/G_MODEL/C/r4/1/gamma*
use_locking(*
T0
Ļ
#generator/G_MODEL/C/r4/1/gamma/readIdentitygenerator/G_MODEL/C/r4/1/gamma*
_output_shapes	
:*1
_class'
%#loc:@generator/G_MODEL/C/r4/1/gamma*
T0

7generator/G_MODEL/C/r4/1/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"         
Ũ
%generator/G_MODEL/C/r4/1/moments/meanMean!generator/G_MODEL/C/r4/r4/BiasAdd7generator/G_MODEL/C/r4/1/moments/mean/reduction_indices*
	keep_dims(*
T0*

Tidx0*&
_output_shapes
:

-generator/G_MODEL/C/r4/1/moments/StopGradientStopGradient%generator/G_MODEL/C/r4/1/moments/mean*&
_output_shapes
:*
T0
Ý
2generator/G_MODEL/C/r4/1/moments/SquaredDifferenceSquaredDifference!generator/G_MODEL/C/r4/r4/BiasAdd-generator/G_MODEL/C/r4/1/moments/StopGradient*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0

;generator/G_MODEL/C/r4/1/moments/variance/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0
ð
)generator/G_MODEL/C/r4/1/moments/varianceMean2generator/G_MODEL/C/r4/1/moments/SquaredDifference;generator/G_MODEL/C/r4/1/moments/variance/reduction_indices*
	keep_dims(*
T0*

Tidx0*&
_output_shapes
:
m
(generator/G_MODEL/C/r4/1/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *Ėž+*
dtype0
ĩ
&generator/G_MODEL/C/r4/1/batchnorm/addAddV2)generator/G_MODEL/C/r4/1/moments/variance(generator/G_MODEL/C/r4/1/batchnorm/add/y*&
_output_shapes
:*
T0

(generator/G_MODEL/C/r4/1/batchnorm/RsqrtRsqrt&generator/G_MODEL/C/r4/1/batchnorm/add*
T0*&
_output_shapes
:
Ū
&generator/G_MODEL/C/r4/1/batchnorm/mulMul(generator/G_MODEL/C/r4/1/batchnorm/Rsqrt#generator/G_MODEL/C/r4/1/gamma/read*
T0*'
_output_shapes
:
ū
(generator/G_MODEL/C/r4/1/batchnorm/mul_1Mul!generator/G_MODEL/C/r4/r4/BiasAdd&generator/G_MODEL/C/r4/1/batchnorm/mul*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
°
(generator/G_MODEL/C/r4/1/batchnorm/mul_2Mul%generator/G_MODEL/C/r4/1/moments/mean&generator/G_MODEL/C/r4/1/batchnorm/mul*'
_output_shapes
:*
T0
­
&generator/G_MODEL/C/r4/1/batchnorm/subSub"generator/G_MODEL/C/r4/1/beta/read(generator/G_MODEL/C/r4/1/batchnorm/mul_2*'
_output_shapes
:*
T0
Į
(generator/G_MODEL/C/r4/1/batchnorm/add_1AddV2(generator/G_MODEL/C/r4/1/batchnorm/mul_1&generator/G_MODEL/C/r4/1/batchnorm/sub*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
­
"generator/G_MODEL/C/r4/LeakyRelu_1	LeakyRelu(generator/G_MODEL/C/r4/1/batchnorm/add_1*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
alpha%ÍĖL>
Û
Hgenerator/G_MODEL/C/r4/Conv_1/weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*8
_class.
,*loc:@generator/G_MODEL/C/r4/Conv_1/weights*%
valueB"            
Æ
Ggenerator/G_MODEL/C/r4/Conv_1/weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *8
_class.
,*loc:@generator/G_MODEL/C/r4/Conv_1/weights*
_output_shapes
: 
Č
Igenerator/G_MODEL/C/r4/Conv_1/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *8
_class.
,*loc:@generator/G_MODEL/C/r4/Conv_1/weights*
valueB
 *Eņ=
Â
Rgenerator/G_MODEL/C/r4/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHgenerator/G_MODEL/C/r4/Conv_1/weights/Initializer/truncated_normal/shape*(
_output_shapes
:*
seed2 *8
_class.
,*loc:@generator/G_MODEL/C/r4/Conv_1/weights*
T0*
dtype0*

seed 
Ų
Fgenerator/G_MODEL/C/r4/Conv_1/weights/Initializer/truncated_normal/mulMulRgenerator/G_MODEL/C/r4/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalIgenerator/G_MODEL/C/r4/Conv_1/weights/Initializer/truncated_normal/stddev*8
_class.
,*loc:@generator/G_MODEL/C/r4/Conv_1/weights*(
_output_shapes
:*
T0
Į
Bgenerator/G_MODEL/C/r4/Conv_1/weights/Initializer/truncated_normalAddFgenerator/G_MODEL/C/r4/Conv_1/weights/Initializer/truncated_normal/mulGgenerator/G_MODEL/C/r4/Conv_1/weights/Initializer/truncated_normal/mean*(
_output_shapes
:*
T0*8
_class.
,*loc:@generator/G_MODEL/C/r4/Conv_1/weights
į
%generator/G_MODEL/C/r4/Conv_1/weights
VariableV2*
shape:*
dtype0*(
_output_shapes
:*
	container *
shared_name *8
_class.
,*loc:@generator/G_MODEL/C/r4/Conv_1/weights
·
,generator/G_MODEL/C/r4/Conv_1/weights/AssignAssign%generator/G_MODEL/C/r4/Conv_1/weightsBgenerator/G_MODEL/C/r4/Conv_1/weights/Initializer/truncated_normal*(
_output_shapes
:*8
_class.
,*loc:@generator/G_MODEL/C/r4/Conv_1/weights*
T0*
use_locking(*
validate_shape(
Ę
*generator/G_MODEL/C/r4/Conv_1/weights/readIdentity%generator/G_MODEL/C/r4/Conv_1/weights*8
_class.
,*loc:@generator/G_MODEL/C/r4/Conv_1/weights*
T0*(
_output_shapes
:
|
+generator/G_MODEL/C/r4/Conv_1/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ę
$generator/G_MODEL/C/r4/Conv_1/Conv2DConv2D"generator/G_MODEL/C/r4/LeakyRelu_1*generator/G_MODEL/C/r4/Conv_1/weights/read*
data_formatNHWC*
	dilations
*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
strides
*
paddingVALID*
use_cudnn_on_gpu(*
explicit_paddings
 
°
/generator/G_MODEL/C/r4/2/beta/Initializer/zerosConst*0
_class&
$"loc:@generator/G_MODEL/C/r4/2/beta*
dtype0*
valueB*    *
_output_shapes	
:
―
generator/G_MODEL/C/r4/2/beta
VariableV2*
dtype0*
_output_shapes	
:*
shape:*
	container *
shared_name *0
_class&
$"loc:@generator/G_MODEL/C/r4/2/beta
ĸ
$generator/G_MODEL/C/r4/2/beta/AssignAssigngenerator/G_MODEL/C/r4/2/beta/generator/G_MODEL/C/r4/2/beta/Initializer/zeros*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(*0
_class&
$"loc:@generator/G_MODEL/C/r4/2/beta
Ĩ
"generator/G_MODEL/C/r4/2/beta/readIdentitygenerator/G_MODEL/C/r4/2/beta*
T0*0
_class&
$"loc:@generator/G_MODEL/C/r4/2/beta*
_output_shapes	
:
ą
/generator/G_MODEL/C/r4/2/gamma/Initializer/onesConst*
valueB*  ?*1
_class'
%#loc:@generator/G_MODEL/C/r4/2/gamma*
dtype0*
_output_shapes	
:
ŋ
generator/G_MODEL/C/r4/2/gamma
VariableV2*
shape:*
_output_shapes	
:*1
_class'
%#loc:@generator/G_MODEL/C/r4/2/gamma*
shared_name *
	container *
dtype0

%generator/G_MODEL/C/r4/2/gamma/AssignAssigngenerator/G_MODEL/C/r4/2/gamma/generator/G_MODEL/C/r4/2/gamma/Initializer/ones*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*1
_class'
%#loc:@generator/G_MODEL/C/r4/2/gamma
Ļ
#generator/G_MODEL/C/r4/2/gamma/readIdentitygenerator/G_MODEL/C/r4/2/gamma*
T0*1
_class'
%#loc:@generator/G_MODEL/C/r4/2/gamma*
_output_shapes	
:

7generator/G_MODEL/C/r4/2/moments/mean/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0
Ú
%generator/G_MODEL/C/r4/2/moments/meanMean$generator/G_MODEL/C/r4/Conv_1/Conv2D7generator/G_MODEL/C/r4/2/moments/mean/reduction_indices*

Tidx0*&
_output_shapes
:*
T0*
	keep_dims(

-generator/G_MODEL/C/r4/2/moments/StopGradientStopGradient%generator/G_MODEL/C/r4/2/moments/mean*
T0*&
_output_shapes
:
ā
2generator/G_MODEL/C/r4/2/moments/SquaredDifferenceSquaredDifference$generator/G_MODEL/C/r4/Conv_1/Conv2D-generator/G_MODEL/C/r4/2/moments/StopGradient*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0

;generator/G_MODEL/C/r4/2/moments/variance/reduction_indicesConst*
dtype0*!
valueB"         *
_output_shapes
:
ð
)generator/G_MODEL/C/r4/2/moments/varianceMean2generator/G_MODEL/C/r4/2/moments/SquaredDifference;generator/G_MODEL/C/r4/2/moments/variance/reduction_indices*
T0*
	keep_dims(*&
_output_shapes
:*

Tidx0
m
(generator/G_MODEL/C/r4/2/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *Ėž+*
dtype0
ĩ
&generator/G_MODEL/C/r4/2/batchnorm/addAddV2)generator/G_MODEL/C/r4/2/moments/variance(generator/G_MODEL/C/r4/2/batchnorm/add/y*&
_output_shapes
:*
T0

(generator/G_MODEL/C/r4/2/batchnorm/RsqrtRsqrt&generator/G_MODEL/C/r4/2/batchnorm/add*&
_output_shapes
:*
T0
Ū
&generator/G_MODEL/C/r4/2/batchnorm/mulMul(generator/G_MODEL/C/r4/2/batchnorm/Rsqrt#generator/G_MODEL/C/r4/2/gamma/read*
T0*'
_output_shapes
:
Á
(generator/G_MODEL/C/r4/2/batchnorm/mul_1Mul$generator/G_MODEL/C/r4/Conv_1/Conv2D&generator/G_MODEL/C/r4/2/batchnorm/mul*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
°
(generator/G_MODEL/C/r4/2/batchnorm/mul_2Mul%generator/G_MODEL/C/r4/2/moments/mean&generator/G_MODEL/C/r4/2/batchnorm/mul*
T0*'
_output_shapes
:
­
&generator/G_MODEL/C/r4/2/batchnorm/subSub"generator/G_MODEL/C/r4/2/beta/read(generator/G_MODEL/C/r4/2/batchnorm/mul_2*'
_output_shapes
:*
T0
Į
(generator/G_MODEL/C/r4/2/batchnorm/add_1AddV2(generator/G_MODEL/C/r4/2/batchnorm/mul_1&generator/G_MODEL/C/r4/2/batchnorm/sub*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
­
generator/G_MODEL/C/r4/addAddV2generator/G_MODEL/C/r3/add(generator/G_MODEL/C/r4/2/batchnorm/add_1*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0

(generator/G_MODEL/C/MirrorPad_1/paddingsConst*9
value0B."                             *
_output_shapes

:*
dtype0
Ú
generator/G_MODEL/C/MirrorPad_1	MirrorPadgenerator/G_MODEL/C/r4/add(generator/G_MODEL/C/MirrorPad_1/paddings*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
mode	REFLECT*
	Tpaddings0*
T0
Õ
Egenerator/G_MODEL/C/Conv_1/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *
dtype0*5
_class+
)'loc:@generator/G_MODEL/C/Conv_1/weights*
_output_shapes
:
Ā
Dgenerator/G_MODEL/C/Conv_1/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *5
_class+
)'loc:@generator/G_MODEL/C/Conv_1/weights*
dtype0
Â
Fgenerator/G_MODEL/C/Conv_1/weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*5
_class+
)'loc:@generator/G_MODEL/C/Conv_1/weights*
valueB
 *	=
đ
Ogenerator/G_MODEL/C/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalEgenerator/G_MODEL/C/Conv_1/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:*
T0*
seed2 *5
_class+
)'loc:@generator/G_MODEL/C/Conv_1/weights*

seed 
Í
Cgenerator/G_MODEL/C/Conv_1/weights/Initializer/truncated_normal/mulMulOgenerator/G_MODEL/C/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalFgenerator/G_MODEL/C/Conv_1/weights/Initializer/truncated_normal/stddev*
T0*(
_output_shapes
:*5
_class+
)'loc:@generator/G_MODEL/C/Conv_1/weights
ŧ
?generator/G_MODEL/C/Conv_1/weights/Initializer/truncated_normalAddCgenerator/G_MODEL/C/Conv_1/weights/Initializer/truncated_normal/mulDgenerator/G_MODEL/C/Conv_1/weights/Initializer/truncated_normal/mean*5
_class+
)'loc:@generator/G_MODEL/C/Conv_1/weights*(
_output_shapes
:*
T0
á
"generator/G_MODEL/C/Conv_1/weights
VariableV2*(
_output_shapes
:*
shared_name *
shape:*
dtype0*5
_class+
)'loc:@generator/G_MODEL/C/Conv_1/weights*
	container 
Ŧ
)generator/G_MODEL/C/Conv_1/weights/AssignAssign"generator/G_MODEL/C/Conv_1/weights?generator/G_MODEL/C/Conv_1/weights/Initializer/truncated_normal*5
_class+
)'loc:@generator/G_MODEL/C/Conv_1/weights*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Á
'generator/G_MODEL/C/Conv_1/weights/readIdentity"generator/G_MODEL/C/Conv_1/weights*
T0*5
_class+
)'loc:@generator/G_MODEL/C/Conv_1/weights*(
_output_shapes
:
y
(generator/G_MODEL/C/Conv_1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Á
!generator/G_MODEL/C/Conv_1/Conv2DConv2Dgenerator/G_MODEL/C/MirrorPad_1'generator/G_MODEL/C/Conv_1/weights/read*
	dilations
*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0*
use_cudnn_on_gpu(*
strides
*
paddingVALID*
explicit_paddings
 *
data_formatNHWC
ū
6generator/G_MODEL/C/LayerNorm_1/beta/Initializer/zerosConst*7
_class-
+)loc:@generator/G_MODEL/C/LayerNorm_1/beta*
_output_shapes	
:*
dtype0*
valueB*    
Ë
$generator/G_MODEL/C/LayerNorm_1/beta
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes	
:*7
_class-
+)loc:@generator/G_MODEL/C/LayerNorm_1/beta

+generator/G_MODEL/C/LayerNorm_1/beta/AssignAssign$generator/G_MODEL/C/LayerNorm_1/beta6generator/G_MODEL/C/LayerNorm_1/beta/Initializer/zeros*
use_locking(*7
_class-
+)loc:@generator/G_MODEL/C/LayerNorm_1/beta*
validate_shape(*
T0*
_output_shapes	
:
š
)generator/G_MODEL/C/LayerNorm_1/beta/readIdentity$generator/G_MODEL/C/LayerNorm_1/beta*
_output_shapes	
:*
T0*7
_class-
+)loc:@generator/G_MODEL/C/LayerNorm_1/beta
ŋ
6generator/G_MODEL/C/LayerNorm_1/gamma/Initializer/onesConst*
valueB*  ?*8
_class.
,*loc:@generator/G_MODEL/C/LayerNorm_1/gamma*
_output_shapes	
:*
dtype0
Í
%generator/G_MODEL/C/LayerNorm_1/gamma
VariableV2*
_output_shapes	
:*8
_class.
,*loc:@generator/G_MODEL/C/LayerNorm_1/gamma*
dtype0*
	container *
shape:*
shared_name 

,generator/G_MODEL/C/LayerNorm_1/gamma/AssignAssign%generator/G_MODEL/C/LayerNorm_1/gamma6generator/G_MODEL/C/LayerNorm_1/gamma/Initializer/ones*
use_locking(*
_output_shapes	
:*8
_class.
,*loc:@generator/G_MODEL/C/LayerNorm_1/gamma*
T0*
validate_shape(
―
*generator/G_MODEL/C/LayerNorm_1/gamma/readIdentity%generator/G_MODEL/C/LayerNorm_1/gamma*
T0*
_output_shapes	
:*8
_class.
,*loc:@generator/G_MODEL/C/LayerNorm_1/gamma

>generator/G_MODEL/C/LayerNorm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
å
,generator/G_MODEL/C/LayerNorm_1/moments/meanMean!generator/G_MODEL/C/Conv_1/Conv2D>generator/G_MODEL/C/LayerNorm_1/moments/mean/reduction_indices*
T0*
	keep_dims(*

Tidx0*&
_output_shapes
:
Ģ
4generator/G_MODEL/C/LayerNorm_1/moments/StopGradientStopGradient,generator/G_MODEL/C/LayerNorm_1/moments/mean*
T0*&
_output_shapes
:
ë
9generator/G_MODEL/C/LayerNorm_1/moments/SquaredDifferenceSquaredDifference!generator/G_MODEL/C/Conv_1/Conv2D4generator/G_MODEL/C/LayerNorm_1/moments/StopGradient*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ

Bgenerator/G_MODEL/C/LayerNorm_1/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"         

0generator/G_MODEL/C/LayerNorm_1/moments/varianceMean9generator/G_MODEL/C/LayerNorm_1/moments/SquaredDifferenceBgenerator/G_MODEL/C/LayerNorm_1/moments/variance/reduction_indices*
	keep_dims(*
T0*&
_output_shapes
:*

Tidx0
t
/generator/G_MODEL/C/LayerNorm_1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ėž+
Ę
-generator/G_MODEL/C/LayerNorm_1/batchnorm/addAddV20generator/G_MODEL/C/LayerNorm_1/moments/variance/generator/G_MODEL/C/LayerNorm_1/batchnorm/add/y*
T0*&
_output_shapes
:

/generator/G_MODEL/C/LayerNorm_1/batchnorm/RsqrtRsqrt-generator/G_MODEL/C/LayerNorm_1/batchnorm/add*
T0*&
_output_shapes
:
Ã
-generator/G_MODEL/C/LayerNorm_1/batchnorm/mulMul/generator/G_MODEL/C/LayerNorm_1/batchnorm/Rsqrt*generator/G_MODEL/C/LayerNorm_1/gamma/read*
T0*'
_output_shapes
:
Ė
/generator/G_MODEL/C/LayerNorm_1/batchnorm/mul_1Mul!generator/G_MODEL/C/Conv_1/Conv2D-generator/G_MODEL/C/LayerNorm_1/batchnorm/mul*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Å
/generator/G_MODEL/C/LayerNorm_1/batchnorm/mul_2Mul,generator/G_MODEL/C/LayerNorm_1/moments/mean-generator/G_MODEL/C/LayerNorm_1/batchnorm/mul*'
_output_shapes
:*
T0
Â
-generator/G_MODEL/C/LayerNorm_1/batchnorm/subSub)generator/G_MODEL/C/LayerNorm_1/beta/read/generator/G_MODEL/C/LayerNorm_1/batchnorm/mul_2*
T0*'
_output_shapes
:
Ü
/generator/G_MODEL/C/LayerNorm_1/batchnorm/add_1AddV2/generator/G_MODEL/C/LayerNorm_1/batchnorm/mul_1-generator/G_MODEL/C/LayerNorm_1/batchnorm/sub*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
ą
generator/G_MODEL/C/LeakyRelu_1	LeakyRelu/generator/G_MODEL/C/LayerNorm_1/batchnorm/add_1*
alpha%ÍĖL>*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
x
generator/G_MODEL/D/ShapeShapegenerator/G_MODEL/C/LeakyRelu_1*
_output_shapes
:*
T0*
out_type0
q
'generator/G_MODEL/D/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
s
)generator/G_MODEL/D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
s
)generator/G_MODEL/D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ý
!generator/G_MODEL/D/strided_sliceStridedSlicegenerator/G_MODEL/D/Shape'generator/G_MODEL/D/strided_slice/stack)generator/G_MODEL/D/strided_slice/stack_1)generator/G_MODEL/D/strided_slice/stack_2*
end_mask *
T0*
Index0*
ellipsis_mask *
_output_shapes
: *
shrink_axis_mask*

begin_mask *
new_axis_mask 
[
generator/G_MODEL/D/mul/xConst*
_output_shapes
: *
value	B :*
dtype0
}
generator/G_MODEL/D/mulMulgenerator/G_MODEL/D/mul/x!generator/G_MODEL/D/strided_slice*
T0*
_output_shapes
: 
z
generator/G_MODEL/D/Shape_1Shapegenerator/G_MODEL/C/LeakyRelu_1*
T0*
out_type0*
_output_shapes
:
s
)generator/G_MODEL/D/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
u
+generator/G_MODEL/D/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
u
+generator/G_MODEL/D/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
į
#generator/G_MODEL/D/strided_slice_1StridedSlicegenerator/G_MODEL/D/Shape_1)generator/G_MODEL/D/strided_slice_1/stack+generator/G_MODEL/D/strided_slice_1/stack_1+generator/G_MODEL/D/strided_slice_1/stack_2*
ellipsis_mask *
shrink_axis_mask*
T0*
Index0*
end_mask *
new_axis_mask *

begin_mask *
_output_shapes
: 
]
generator/G_MODEL/D/mul_1/xConst*
value	B :*
_output_shapes
: *
dtype0

generator/G_MODEL/D/mul_1Mulgenerator/G_MODEL/D/mul_1/x#generator/G_MODEL/D/strided_slice_1*
_output_shapes
: *
T0

generator/G_MODEL/D/resize/sizePackgenerator/G_MODEL/D/mulgenerator/G_MODEL/D/mul_1*

axis *
_output_shapes
:*
T0*
N
ð
)generator/G_MODEL/D/resize/ResizeBilinearResizeBilineargenerator/G_MODEL/C/LeakyRelu_1generator/G_MODEL/D/resize/size*
T0*
half_pixel_centers( *
align_corners( *9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ

&generator/G_MODEL/D/MirrorPad/paddingsConst*9
value0B."                             *
_output_shapes

:*
dtype0
å
generator/G_MODEL/D/MirrorPad	MirrorPad)generator/G_MODEL/D/resize/ResizeBilinear&generator/G_MODEL/D/MirrorPad/paddings*
	Tpaddings0*
mode	REFLECT*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Ņ
Cgenerator/G_MODEL/D/Conv/weights/Initializer/truncated_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:*3
_class)
'%loc:@generator/G_MODEL/D/Conv/weights
ž
Bgenerator/G_MODEL/D/Conv/weights/Initializer/truncated_normal/meanConst*3
_class)
'%loc:@generator/G_MODEL/D/Conv/weights*
_output_shapes
: *
valueB
 *    *
dtype0
ū
Dgenerator/G_MODEL/D/Conv/weights/Initializer/truncated_normal/stddevConst*
dtype0*3
_class)
'%loc:@generator/G_MODEL/D/Conv/weights*
valueB
 *B=*
_output_shapes
: 
ģ
Mgenerator/G_MODEL/D/Conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalCgenerator/G_MODEL/D/Conv/weights/Initializer/truncated_normal/shape*
T0*
seed2 *

seed *3
_class)
'%loc:@generator/G_MODEL/D/Conv/weights*
dtype0*(
_output_shapes
:
Å
Agenerator/G_MODEL/D/Conv/weights/Initializer/truncated_normal/mulMulMgenerator/G_MODEL/D/Conv/weights/Initializer/truncated_normal/TruncatedNormalDgenerator/G_MODEL/D/Conv/weights/Initializer/truncated_normal/stddev*3
_class)
'%loc:@generator/G_MODEL/D/Conv/weights*(
_output_shapes
:*
T0
ģ
=generator/G_MODEL/D/Conv/weights/Initializer/truncated_normalAddAgenerator/G_MODEL/D/Conv/weights/Initializer/truncated_normal/mulBgenerator/G_MODEL/D/Conv/weights/Initializer/truncated_normal/mean*
T0*(
_output_shapes
:*3
_class)
'%loc:@generator/G_MODEL/D/Conv/weights
Ý
 generator/G_MODEL/D/Conv/weights
VariableV2*(
_output_shapes
:*3
_class)
'%loc:@generator/G_MODEL/D/Conv/weights*
dtype0*
	container *
shared_name *
shape:
Ģ
'generator/G_MODEL/D/Conv/weights/AssignAssign generator/G_MODEL/D/Conv/weights=generator/G_MODEL/D/Conv/weights/Initializer/truncated_normal*3
_class)
'%loc:@generator/G_MODEL/D/Conv/weights*(
_output_shapes
:*
T0*
use_locking(*
validate_shape(
ŧ
%generator/G_MODEL/D/Conv/weights/readIdentity generator/G_MODEL/D/Conv/weights*(
_output_shapes
:*
T0*3
_class)
'%loc:@generator/G_MODEL/D/Conv/weights
w
&generator/G_MODEL/D/Conv/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
ŧ
generator/G_MODEL/D/Conv/Conv2DConv2Dgenerator/G_MODEL/D/MirrorPad%generator/G_MODEL/D/Conv/weights/read*
explicit_paddings
 *
	dilations
*
paddingVALID*
use_cudnn_on_gpu(*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0*
strides
*
data_formatNHWC
š
4generator/G_MODEL/D/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:*5
_class+
)'loc:@generator/G_MODEL/D/LayerNorm/beta*
valueB*    
Į
"generator/G_MODEL/D/LayerNorm/beta
VariableV2*
_output_shapes	
:*
	container *5
_class+
)'loc:@generator/G_MODEL/D/LayerNorm/beta*
shape:*
dtype0*
shared_name 

)generator/G_MODEL/D/LayerNorm/beta/AssignAssign"generator/G_MODEL/D/LayerNorm/beta4generator/G_MODEL/D/LayerNorm/beta/Initializer/zeros*
T0*
use_locking(*
_output_shapes	
:*5
_class+
)'loc:@generator/G_MODEL/D/LayerNorm/beta*
validate_shape(
ī
'generator/G_MODEL/D/LayerNorm/beta/readIdentity"generator/G_MODEL/D/LayerNorm/beta*5
_class+
)'loc:@generator/G_MODEL/D/LayerNorm/beta*
T0*
_output_shapes	
:
ŧ
4generator/G_MODEL/D/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*
dtype0*6
_class,
*(loc:@generator/G_MODEL/D/LayerNorm/gamma*
_output_shapes	
:
É
#generator/G_MODEL/D/LayerNorm/gamma
VariableV2*
shared_name *
_output_shapes	
:*
dtype0*
	container *6
_class,
*(loc:@generator/G_MODEL/D/LayerNorm/gamma*
shape:

*generator/G_MODEL/D/LayerNorm/gamma/AssignAssign#generator/G_MODEL/D/LayerNorm/gamma4generator/G_MODEL/D/LayerNorm/gamma/Initializer/ones*
validate_shape(*
T0*6
_class,
*(loc:@generator/G_MODEL/D/LayerNorm/gamma*
use_locking(*
_output_shapes	
:
·
(generator/G_MODEL/D/LayerNorm/gamma/readIdentity#generator/G_MODEL/D/LayerNorm/gamma*
_output_shapes	
:*6
_class,
*(loc:@generator/G_MODEL/D/LayerNorm/gamma*
T0

<generator/G_MODEL/D/LayerNorm/moments/mean/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0
ß
*generator/G_MODEL/D/LayerNorm/moments/meanMeangenerator/G_MODEL/D/Conv/Conv2D<generator/G_MODEL/D/LayerNorm/moments/mean/reduction_indices*&
_output_shapes
:*
	keep_dims(*

Tidx0*
T0

2generator/G_MODEL/D/LayerNorm/moments/StopGradientStopGradient*generator/G_MODEL/D/LayerNorm/moments/mean*
T0*&
_output_shapes
:
å
7generator/G_MODEL/D/LayerNorm/moments/SquaredDifferenceSquaredDifferencegenerator/G_MODEL/D/Conv/Conv2D2generator/G_MODEL/D/LayerNorm/moments/StopGradient*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ

@generator/G_MODEL/D/LayerNorm/moments/variance/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0
ĸ
.generator/G_MODEL/D/LayerNorm/moments/varianceMean7generator/G_MODEL/D/LayerNorm/moments/SquaredDifference@generator/G_MODEL/D/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*&
_output_shapes
:*
T0
r
-generator/G_MODEL/D/LayerNorm/batchnorm/add/yConst*
valueB
 *Ėž+*
_output_shapes
: *
dtype0
Ä
+generator/G_MODEL/D/LayerNorm/batchnorm/addAddV2.generator/G_MODEL/D/LayerNorm/moments/variance-generator/G_MODEL/D/LayerNorm/batchnorm/add/y*
T0*&
_output_shapes
:

-generator/G_MODEL/D/LayerNorm/batchnorm/RsqrtRsqrt+generator/G_MODEL/D/LayerNorm/batchnorm/add*
T0*&
_output_shapes
:
―
+generator/G_MODEL/D/LayerNorm/batchnorm/mulMul-generator/G_MODEL/D/LayerNorm/batchnorm/Rsqrt(generator/G_MODEL/D/LayerNorm/gamma/read*'
_output_shapes
:*
T0
Æ
-generator/G_MODEL/D/LayerNorm/batchnorm/mul_1Mulgenerator/G_MODEL/D/Conv/Conv2D+generator/G_MODEL/D/LayerNorm/batchnorm/mul*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
ŋ
-generator/G_MODEL/D/LayerNorm/batchnorm/mul_2Mul*generator/G_MODEL/D/LayerNorm/moments/mean+generator/G_MODEL/D/LayerNorm/batchnorm/mul*
T0*'
_output_shapes
:
ž
+generator/G_MODEL/D/LayerNorm/batchnorm/subSub'generator/G_MODEL/D/LayerNorm/beta/read-generator/G_MODEL/D/LayerNorm/batchnorm/mul_2*
T0*'
_output_shapes
:
Ö
-generator/G_MODEL/D/LayerNorm/batchnorm/add_1AddV2-generator/G_MODEL/D/LayerNorm/batchnorm/mul_1+generator/G_MODEL/D/LayerNorm/batchnorm/sub*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
­
generator/G_MODEL/D/LeakyRelu	LeakyRelu-generator/G_MODEL/D/LayerNorm/batchnorm/add_1*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
alpha%ÍĖL>*
T0

(generator/G_MODEL/D/MirrorPad_1/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
Ý
generator/G_MODEL/D/MirrorPad_1	MirrorPadgenerator/G_MODEL/D/LeakyRelu(generator/G_MODEL/D/MirrorPad_1/paddings*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0*
	Tpaddings0*
mode	REFLECT
Õ
Egenerator/G_MODEL/D/Conv_1/weights/Initializer/truncated_normal/shapeConst*5
_class+
)'loc:@generator/G_MODEL/D/Conv_1/weights*%
valueB"            *
_output_shapes
:*
dtype0
Ā
Dgenerator/G_MODEL/D/Conv_1/weights/Initializer/truncated_normal/meanConst*5
_class+
)'loc:@generator/G_MODEL/D/Conv_1/weights*
_output_shapes
: *
valueB
 *    *
dtype0
Â
Fgenerator/G_MODEL/D/Conv_1/weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *5
_class+
)'loc:@generator/G_MODEL/D/Conv_1/weights*
valueB
 *B=*
dtype0
đ
Ogenerator/G_MODEL/D/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalEgenerator/G_MODEL/D/Conv_1/weights/Initializer/truncated_normal/shape*
T0*
dtype0*5
_class+
)'loc:@generator/G_MODEL/D/Conv_1/weights*(
_output_shapes
:*
seed2 *

seed 
Í
Cgenerator/G_MODEL/D/Conv_1/weights/Initializer/truncated_normal/mulMulOgenerator/G_MODEL/D/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalFgenerator/G_MODEL/D/Conv_1/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:*
T0*5
_class+
)'loc:@generator/G_MODEL/D/Conv_1/weights
ŧ
?generator/G_MODEL/D/Conv_1/weights/Initializer/truncated_normalAddCgenerator/G_MODEL/D/Conv_1/weights/Initializer/truncated_normal/mulDgenerator/G_MODEL/D/Conv_1/weights/Initializer/truncated_normal/mean*(
_output_shapes
:*
T0*5
_class+
)'loc:@generator/G_MODEL/D/Conv_1/weights
á
"generator/G_MODEL/D/Conv_1/weights
VariableV2*
	container *
dtype0*5
_class+
)'loc:@generator/G_MODEL/D/Conv_1/weights*
shape:*(
_output_shapes
:*
shared_name 
Ŧ
)generator/G_MODEL/D/Conv_1/weights/AssignAssign"generator/G_MODEL/D/Conv_1/weights?generator/G_MODEL/D/Conv_1/weights/Initializer/truncated_normal*
T0*5
_class+
)'loc:@generator/G_MODEL/D/Conv_1/weights*
use_locking(*(
_output_shapes
:*
validate_shape(
Á
'generator/G_MODEL/D/Conv_1/weights/readIdentity"generator/G_MODEL/D/Conv_1/weights*
T0*(
_output_shapes
:*5
_class+
)'loc:@generator/G_MODEL/D/Conv_1/weights
y
(generator/G_MODEL/D/Conv_1/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Á
!generator/G_MODEL/D/Conv_1/Conv2DConv2Dgenerator/G_MODEL/D/MirrorPad_1'generator/G_MODEL/D/Conv_1/weights/read*
explicit_paddings
 *9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0*
	dilations
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*
strides

ū
6generator/G_MODEL/D/LayerNorm_1/beta/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *7
_class-
+)loc:@generator/G_MODEL/D/LayerNorm_1/beta*
dtype0
Ë
$generator/G_MODEL/D/LayerNorm_1/beta
VariableV2*
shared_name *7
_class-
+)loc:@generator/G_MODEL/D/LayerNorm_1/beta*
	container *
dtype0*
shape:*
_output_shapes	
:

+generator/G_MODEL/D/LayerNorm_1/beta/AssignAssign$generator/G_MODEL/D/LayerNorm_1/beta6generator/G_MODEL/D/LayerNorm_1/beta/Initializer/zeros*
T0*7
_class-
+)loc:@generator/G_MODEL/D/LayerNorm_1/beta*
validate_shape(*
use_locking(*
_output_shapes	
:
š
)generator/G_MODEL/D/LayerNorm_1/beta/readIdentity$generator/G_MODEL/D/LayerNorm_1/beta*
_output_shapes	
:*7
_class-
+)loc:@generator/G_MODEL/D/LayerNorm_1/beta*
T0
ŋ
6generator/G_MODEL/D/LayerNorm_1/gamma/Initializer/onesConst*
dtype0*8
_class.
,*loc:@generator/G_MODEL/D/LayerNorm_1/gamma*
valueB*  ?*
_output_shapes	
:
Í
%generator/G_MODEL/D/LayerNorm_1/gamma
VariableV2*
	container *8
_class.
,*loc:@generator/G_MODEL/D/LayerNorm_1/gamma*
shared_name *
dtype0*
_output_shapes	
:*
shape:

,generator/G_MODEL/D/LayerNorm_1/gamma/AssignAssign%generator/G_MODEL/D/LayerNorm_1/gamma6generator/G_MODEL/D/LayerNorm_1/gamma/Initializer/ones*
use_locking(*
_output_shapes	
:*
T0*8
_class.
,*loc:@generator/G_MODEL/D/LayerNorm_1/gamma*
validate_shape(
―
*generator/G_MODEL/D/LayerNorm_1/gamma/readIdentity%generator/G_MODEL/D/LayerNorm_1/gamma*
_output_shapes	
:*8
_class.
,*loc:@generator/G_MODEL/D/LayerNorm_1/gamma*
T0

>generator/G_MODEL/D/LayerNorm_1/moments/mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:
å
,generator/G_MODEL/D/LayerNorm_1/moments/meanMean!generator/G_MODEL/D/Conv_1/Conv2D>generator/G_MODEL/D/LayerNorm_1/moments/mean/reduction_indices*
	keep_dims(*
T0*

Tidx0*&
_output_shapes
:
Ģ
4generator/G_MODEL/D/LayerNorm_1/moments/StopGradientStopGradient,generator/G_MODEL/D/LayerNorm_1/moments/mean*&
_output_shapes
:*
T0
ë
9generator/G_MODEL/D/LayerNorm_1/moments/SquaredDifferenceSquaredDifference!generator/G_MODEL/D/Conv_1/Conv2D4generator/G_MODEL/D/LayerNorm_1/moments/StopGradient*
T0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ

Bgenerator/G_MODEL/D/LayerNorm_1/moments/variance/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0

0generator/G_MODEL/D/LayerNorm_1/moments/varianceMean9generator/G_MODEL/D/LayerNorm_1/moments/SquaredDifferenceBgenerator/G_MODEL/D/LayerNorm_1/moments/variance/reduction_indices*&
_output_shapes
:*
	keep_dims(*

Tidx0*
T0
t
/generator/G_MODEL/D/LayerNorm_1/batchnorm/add/yConst*
valueB
 *Ėž+*
dtype0*
_output_shapes
: 
Ę
-generator/G_MODEL/D/LayerNorm_1/batchnorm/addAddV20generator/G_MODEL/D/LayerNorm_1/moments/variance/generator/G_MODEL/D/LayerNorm_1/batchnorm/add/y*
T0*&
_output_shapes
:

/generator/G_MODEL/D/LayerNorm_1/batchnorm/RsqrtRsqrt-generator/G_MODEL/D/LayerNorm_1/batchnorm/add*&
_output_shapes
:*
T0
Ã
-generator/G_MODEL/D/LayerNorm_1/batchnorm/mulMul/generator/G_MODEL/D/LayerNorm_1/batchnorm/Rsqrt*generator/G_MODEL/D/LayerNorm_1/gamma/read*
T0*'
_output_shapes
:
Ė
/generator/G_MODEL/D/LayerNorm_1/batchnorm/mul_1Mul!generator/G_MODEL/D/Conv_1/Conv2D-generator/G_MODEL/D/LayerNorm_1/batchnorm/mul*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
Å
/generator/G_MODEL/D/LayerNorm_1/batchnorm/mul_2Mul,generator/G_MODEL/D/LayerNorm_1/moments/mean-generator/G_MODEL/D/LayerNorm_1/batchnorm/mul*
T0*'
_output_shapes
:
Â
-generator/G_MODEL/D/LayerNorm_1/batchnorm/subSub)generator/G_MODEL/D/LayerNorm_1/beta/read/generator/G_MODEL/D/LayerNorm_1/batchnorm/mul_2*'
_output_shapes
:*
T0
Ü
/generator/G_MODEL/D/LayerNorm_1/batchnorm/add_1AddV2/generator/G_MODEL/D/LayerNorm_1/batchnorm/mul_1-generator/G_MODEL/D/LayerNorm_1/batchnorm/sub*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
ą
generator/G_MODEL/D/LeakyRelu_1	LeakyRelu/generator/G_MODEL/D/LayerNorm_1/batchnorm/add_1*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
alpha%ÍĖL>*
T0
x
generator/G_MODEL/E/ShapeShapegenerator/G_MODEL/D/LeakyRelu_1*
_output_shapes
:*
out_type0*
T0
q
'generator/G_MODEL/E/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
s
)generator/G_MODEL/E/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
s
)generator/G_MODEL/E/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ý
!generator/G_MODEL/E/strided_sliceStridedSlicegenerator/G_MODEL/E/Shape'generator/G_MODEL/E/strided_slice/stack)generator/G_MODEL/E/strided_slice/stack_1)generator/G_MODEL/E/strided_slice/stack_2*
_output_shapes
: *
new_axis_mask *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *
end_mask *

begin_mask 
[
generator/G_MODEL/E/mul/xConst*
dtype0*
_output_shapes
: *
value	B :
}
generator/G_MODEL/E/mulMulgenerator/G_MODEL/E/mul/x!generator/G_MODEL/E/strided_slice*
_output_shapes
: *
T0
z
generator/G_MODEL/E/Shape_1Shapegenerator/G_MODEL/D/LeakyRelu_1*
_output_shapes
:*
T0*
out_type0
s
)generator/G_MODEL/E/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
u
+generator/G_MODEL/E/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
u
+generator/G_MODEL/E/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
į
#generator/G_MODEL/E/strided_slice_1StridedSlicegenerator/G_MODEL/E/Shape_1)generator/G_MODEL/E/strided_slice_1/stack+generator/G_MODEL/E/strided_slice_1/stack_1+generator/G_MODEL/E/strided_slice_1/stack_2*
new_axis_mask *
_output_shapes
: *
T0*
ellipsis_mask *

begin_mask *
end_mask *
Index0*
shrink_axis_mask
]
generator/G_MODEL/E/mul_1/xConst*
_output_shapes
: *
value	B :*
dtype0

generator/G_MODEL/E/mul_1Mulgenerator/G_MODEL/E/mul_1/x#generator/G_MODEL/E/strided_slice_1*
T0*
_output_shapes
: 

generator/G_MODEL/E/resize/sizePackgenerator/G_MODEL/E/mulgenerator/G_MODEL/E/mul_1*
_output_shapes
:*
T0*

axis *
N
ð
)generator/G_MODEL/E/resize/ResizeBilinearResizeBilineargenerator/G_MODEL/D/LeakyRelu_1generator/G_MODEL/E/resize/size*
half_pixel_centers( *9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0*
align_corners( 

&generator/G_MODEL/E/MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
å
generator/G_MODEL/E/MirrorPad	MirrorPad)generator/G_MODEL/E/resize/ResizeBilinear&generator/G_MODEL/E/MirrorPad/paddings*
T0*
mode	REFLECT*
	Tpaddings0*9
_output_shapes'
%:#ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Ņ
Cgenerator/G_MODEL/E/Conv/weights/Initializer/truncated_normal/shapeConst*
dtype0*%
valueB"         @   *
_output_shapes
:*3
_class)
'%loc:@generator/G_MODEL/E/Conv/weights
ž
Bgenerator/G_MODEL/E/Conv/weights/Initializer/truncated_normal/meanConst*
_output_shapes
: *3
_class)
'%loc:@generator/G_MODEL/E/Conv/weights*
dtype0*
valueB
 *    
ū
Dgenerator/G_MODEL/E/Conv/weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*3
_class)
'%loc:@generator/G_MODEL/E/Conv/weights*
valueB
 *B=
ē
Mgenerator/G_MODEL/E/Conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalCgenerator/G_MODEL/E/Conv/weights/Initializer/truncated_normal/shape*
seed2 *

seed *3
_class)
'%loc:@generator/G_MODEL/E/Conv/weights*
dtype0*
T0*'
_output_shapes
:@
Ä
Agenerator/G_MODEL/E/Conv/weights/Initializer/truncated_normal/mulMulMgenerator/G_MODEL/E/Conv/weights/Initializer/truncated_normal/TruncatedNormalDgenerator/G_MODEL/E/Conv/weights/Initializer/truncated_normal/stddev*3
_class)
'%loc:@generator/G_MODEL/E/Conv/weights*'
_output_shapes
:@*
T0
ē
=generator/G_MODEL/E/Conv/weights/Initializer/truncated_normalAddAgenerator/G_MODEL/E/Conv/weights/Initializer/truncated_normal/mulBgenerator/G_MODEL/E/Conv/weights/Initializer/truncated_normal/mean*
T0*3
_class)
'%loc:@generator/G_MODEL/E/Conv/weights*'
_output_shapes
:@
Û
 generator/G_MODEL/E/Conv/weights
VariableV2*'
_output_shapes
:@*
dtype0*
shared_name *
shape:@*3
_class)
'%loc:@generator/G_MODEL/E/Conv/weights*
	container 
Ē
'generator/G_MODEL/E/Conv/weights/AssignAssign generator/G_MODEL/E/Conv/weights=generator/G_MODEL/E/Conv/weights/Initializer/truncated_normal*3
_class)
'%loc:@generator/G_MODEL/E/Conv/weights*
T0*
validate_shape(*
use_locking(*'
_output_shapes
:@
š
%generator/G_MODEL/E/Conv/weights/readIdentity generator/G_MODEL/E/Conv/weights*
T0*3
_class)
'%loc:@generator/G_MODEL/E/Conv/weights*'
_output_shapes
:@
w
&generator/G_MODEL/E/Conv/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
š
generator/G_MODEL/E/Conv/Conv2DConv2Dgenerator/G_MODEL/E/MirrorPad%generator/G_MODEL/E/Conv/weights/read*
explicit_paddings
 *8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
data_formatNHWC*
T0*
paddingVALID*
strides
*
	dilations
*
use_cudnn_on_gpu(
ļ
4generator/G_MODEL/E/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:@*5
_class+
)'loc:@generator/G_MODEL/E/LayerNorm/beta*
valueB@*    
Å
"generator/G_MODEL/E/LayerNorm/beta
VariableV2*
shared_name *
	container *
_output_shapes
:@*5
_class+
)'loc:@generator/G_MODEL/E/LayerNorm/beta*
shape:@*
dtype0

)generator/G_MODEL/E/LayerNorm/beta/AssignAssign"generator/G_MODEL/E/LayerNorm/beta4generator/G_MODEL/E/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(*5
_class+
)'loc:@generator/G_MODEL/E/LayerNorm/beta
ģ
'generator/G_MODEL/E/LayerNorm/beta/readIdentity"generator/G_MODEL/E/LayerNorm/beta*
T0*
_output_shapes
:@*5
_class+
)'loc:@generator/G_MODEL/E/LayerNorm/beta
đ
4generator/G_MODEL/E/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:@*6
_class,
*(loc:@generator/G_MODEL/E/LayerNorm/gamma*
valueB@*  ?
Į
#generator/G_MODEL/E/LayerNorm/gamma
VariableV2*
	container *
dtype0*
shared_name *
shape:@*
_output_shapes
:@*6
_class,
*(loc:@generator/G_MODEL/E/LayerNorm/gamma

*generator/G_MODEL/E/LayerNorm/gamma/AssignAssign#generator/G_MODEL/E/LayerNorm/gamma4generator/G_MODEL/E/LayerNorm/gamma/Initializer/ones*
T0*
_output_shapes
:@*6
_class,
*(loc:@generator/G_MODEL/E/LayerNorm/gamma*
validate_shape(*
use_locking(
ķ
(generator/G_MODEL/E/LayerNorm/gamma/readIdentity#generator/G_MODEL/E/LayerNorm/gamma*6
_class,
*(loc:@generator/G_MODEL/E/LayerNorm/gamma*
T0*
_output_shapes
:@

<generator/G_MODEL/E/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
ß
*generator/G_MODEL/E/LayerNorm/moments/meanMeangenerator/G_MODEL/E/Conv/Conv2D<generator/G_MODEL/E/LayerNorm/moments/mean/reduction_indices*
T0*
	keep_dims(*&
_output_shapes
:*

Tidx0

2generator/G_MODEL/E/LayerNorm/moments/StopGradientStopGradient*generator/G_MODEL/E/LayerNorm/moments/mean*&
_output_shapes
:*
T0
ä
7generator/G_MODEL/E/LayerNorm/moments/SquaredDifferenceSquaredDifferencegenerator/G_MODEL/E/Conv/Conv2D2generator/G_MODEL/E/LayerNorm/moments/StopGradient*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
T0

@generator/G_MODEL/E/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
ĸ
.generator/G_MODEL/E/LayerNorm/moments/varianceMean7generator/G_MODEL/E/LayerNorm/moments/SquaredDifference@generator/G_MODEL/E/LayerNorm/moments/variance/reduction_indices*

Tidx0*
T0*
	keep_dims(*&
_output_shapes
:
r
-generator/G_MODEL/E/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ėž+
Ä
+generator/G_MODEL/E/LayerNorm/batchnorm/addAddV2.generator/G_MODEL/E/LayerNorm/moments/variance-generator/G_MODEL/E/LayerNorm/batchnorm/add/y*
T0*&
_output_shapes
:

-generator/G_MODEL/E/LayerNorm/batchnorm/RsqrtRsqrt+generator/G_MODEL/E/LayerNorm/batchnorm/add*
T0*&
_output_shapes
:
ž
+generator/G_MODEL/E/LayerNorm/batchnorm/mulMul-generator/G_MODEL/E/LayerNorm/batchnorm/Rsqrt(generator/G_MODEL/E/LayerNorm/gamma/read*
T0*&
_output_shapes
:@
Å
-generator/G_MODEL/E/LayerNorm/batchnorm/mul_1Mulgenerator/G_MODEL/E/Conv/Conv2D+generator/G_MODEL/E/LayerNorm/batchnorm/mul*
T0*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@
ū
-generator/G_MODEL/E/LayerNorm/batchnorm/mul_2Mul*generator/G_MODEL/E/LayerNorm/moments/mean+generator/G_MODEL/E/LayerNorm/batchnorm/mul*&
_output_shapes
:@*
T0
ŧ
+generator/G_MODEL/E/LayerNorm/batchnorm/subSub'generator/G_MODEL/E/LayerNorm/beta/read-generator/G_MODEL/E/LayerNorm/batchnorm/mul_2*
T0*&
_output_shapes
:@
Õ
-generator/G_MODEL/E/LayerNorm/batchnorm/add_1AddV2-generator/G_MODEL/E/LayerNorm/batchnorm/mul_1+generator/G_MODEL/E/LayerNorm/batchnorm/sub*
T0*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@
Ž
generator/G_MODEL/E/LeakyRelu	LeakyRelu-generator/G_MODEL/E/LayerNorm/batchnorm/add_1*
alpha%ÍĖL>*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
T0

(generator/G_MODEL/E/MirrorPad_1/paddingsConst*9
value0B."                             *
_output_shapes

:*
dtype0
Ü
generator/G_MODEL/E/MirrorPad_1	MirrorPadgenerator/G_MODEL/E/LeakyRelu(generator/G_MODEL/E/MirrorPad_1/paddings*
	Tpaddings0*
T0*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
mode	REFLECT
Õ
Egenerator/G_MODEL/E/Conv_1/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*5
_class+
)'loc:@generator/G_MODEL/E/Conv_1/weights*%
valueB"      @   @   
Ā
Dgenerator/G_MODEL/E/Conv_1/weights/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *5
_class+
)'loc:@generator/G_MODEL/E/Conv_1/weights*
dtype0
Â
Fgenerator/G_MODEL/E/Conv_1/weights/Initializer/truncated_normal/stddevConst*
valueB
 *=*
dtype0*5
_class+
)'loc:@generator/G_MODEL/E/Conv_1/weights*
_output_shapes
: 
·
Ogenerator/G_MODEL/E/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalEgenerator/G_MODEL/E/Conv_1/weights/Initializer/truncated_normal/shape*
T0*

seed *5
_class+
)'loc:@generator/G_MODEL/E/Conv_1/weights*
dtype0*
seed2 *&
_output_shapes
:@@
Ë
Cgenerator/G_MODEL/E/Conv_1/weights/Initializer/truncated_normal/mulMulOgenerator/G_MODEL/E/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalFgenerator/G_MODEL/E/Conv_1/weights/Initializer/truncated_normal/stddev*&
_output_shapes
:@@*
T0*5
_class+
)'loc:@generator/G_MODEL/E/Conv_1/weights
đ
?generator/G_MODEL/E/Conv_1/weights/Initializer/truncated_normalAddCgenerator/G_MODEL/E/Conv_1/weights/Initializer/truncated_normal/mulDgenerator/G_MODEL/E/Conv_1/weights/Initializer/truncated_normal/mean*5
_class+
)'loc:@generator/G_MODEL/E/Conv_1/weights*&
_output_shapes
:@@*
T0
Ý
"generator/G_MODEL/E/Conv_1/weights
VariableV2*
shape:@@*
	container *
shared_name *
dtype0*5
_class+
)'loc:@generator/G_MODEL/E/Conv_1/weights*&
_output_shapes
:@@
Đ
)generator/G_MODEL/E/Conv_1/weights/AssignAssign"generator/G_MODEL/E/Conv_1/weights?generator/G_MODEL/E/Conv_1/weights/Initializer/truncated_normal*5
_class+
)'loc:@generator/G_MODEL/E/Conv_1/weights*
use_locking(*
validate_shape(*&
_output_shapes
:@@*
T0
ŋ
'generator/G_MODEL/E/Conv_1/weights/readIdentity"generator/G_MODEL/E/Conv_1/weights*&
_output_shapes
:@@*5
_class+
)'loc:@generator/G_MODEL/E/Conv_1/weights*
T0
y
(generator/G_MODEL/E/Conv_1/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ā
!generator/G_MODEL/E/Conv_1/Conv2DConv2Dgenerator/G_MODEL/E/MirrorPad_1'generator/G_MODEL/E/Conv_1/weights/read*
paddingVALID*
explicit_paddings
 *
data_formatNHWC*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
use_cudnn_on_gpu(*
	dilations
*
strides
*
T0
ž
6generator/G_MODEL/E/LayerNorm_1/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *7
_class-
+)loc:@generator/G_MODEL/E/LayerNorm_1/beta
É
$generator/G_MODEL/E/LayerNorm_1/beta
VariableV2*7
_class-
+)loc:@generator/G_MODEL/E/LayerNorm_1/beta*
shared_name *
shape:@*
dtype0*
_output_shapes
:@*
	container 

+generator/G_MODEL/E/LayerNorm_1/beta/AssignAssign$generator/G_MODEL/E/LayerNorm_1/beta6generator/G_MODEL/E/LayerNorm_1/beta/Initializer/zeros*
use_locking(*7
_class-
+)loc:@generator/G_MODEL/E/LayerNorm_1/beta*
_output_shapes
:@*
validate_shape(*
T0
đ
)generator/G_MODEL/E/LayerNorm_1/beta/readIdentity$generator/G_MODEL/E/LayerNorm_1/beta*
T0*7
_class-
+)loc:@generator/G_MODEL/E/LayerNorm_1/beta*
_output_shapes
:@
―
6generator/G_MODEL/E/LayerNorm_1/gamma/Initializer/onesConst*
dtype0*8
_class.
,*loc:@generator/G_MODEL/E/LayerNorm_1/gamma*
valueB@*  ?*
_output_shapes
:@
Ë
%generator/G_MODEL/E/LayerNorm_1/gamma
VariableV2*
dtype0*
_output_shapes
:@*
shape:@*8
_class.
,*loc:@generator/G_MODEL/E/LayerNorm_1/gamma*
shared_name *
	container 

,generator/G_MODEL/E/LayerNorm_1/gamma/AssignAssign%generator/G_MODEL/E/LayerNorm_1/gamma6generator/G_MODEL/E/LayerNorm_1/gamma/Initializer/ones*
T0*8
_class.
,*loc:@generator/G_MODEL/E/LayerNorm_1/gamma*
use_locking(*
validate_shape(*
_output_shapes
:@
ž
*generator/G_MODEL/E/LayerNorm_1/gamma/readIdentity%generator/G_MODEL/E/LayerNorm_1/gamma*
T0*
_output_shapes
:@*8
_class.
,*loc:@generator/G_MODEL/E/LayerNorm_1/gamma

>generator/G_MODEL/E/LayerNorm_1/moments/mean/reduction_indicesConst*
dtype0*!
valueB"         *
_output_shapes
:
å
,generator/G_MODEL/E/LayerNorm_1/moments/meanMean!generator/G_MODEL/E/Conv_1/Conv2D>generator/G_MODEL/E/LayerNorm_1/moments/mean/reduction_indices*
T0*

Tidx0*&
_output_shapes
:*
	keep_dims(
Ģ
4generator/G_MODEL/E/LayerNorm_1/moments/StopGradientStopGradient,generator/G_MODEL/E/LayerNorm_1/moments/mean*
T0*&
_output_shapes
:
ę
9generator/G_MODEL/E/LayerNorm_1/moments/SquaredDifferenceSquaredDifference!generator/G_MODEL/E/Conv_1/Conv2D4generator/G_MODEL/E/LayerNorm_1/moments/StopGradient*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
T0

Bgenerator/G_MODEL/E/LayerNorm_1/moments/variance/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0

0generator/G_MODEL/E/LayerNorm_1/moments/varianceMean9generator/G_MODEL/E/LayerNorm_1/moments/SquaredDifferenceBgenerator/G_MODEL/E/LayerNorm_1/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*&
_output_shapes
:*
T0
t
/generator/G_MODEL/E/LayerNorm_1/batchnorm/add/yConst*
valueB
 *Ėž+*
dtype0*
_output_shapes
: 
Ę
-generator/G_MODEL/E/LayerNorm_1/batchnorm/addAddV20generator/G_MODEL/E/LayerNorm_1/moments/variance/generator/G_MODEL/E/LayerNorm_1/batchnorm/add/y*
T0*&
_output_shapes
:

/generator/G_MODEL/E/LayerNorm_1/batchnorm/RsqrtRsqrt-generator/G_MODEL/E/LayerNorm_1/batchnorm/add*&
_output_shapes
:*
T0
Â
-generator/G_MODEL/E/LayerNorm_1/batchnorm/mulMul/generator/G_MODEL/E/LayerNorm_1/batchnorm/Rsqrt*generator/G_MODEL/E/LayerNorm_1/gamma/read*
T0*&
_output_shapes
:@
Ë
/generator/G_MODEL/E/LayerNorm_1/batchnorm/mul_1Mul!generator/G_MODEL/E/Conv_1/Conv2D-generator/G_MODEL/E/LayerNorm_1/batchnorm/mul*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
T0
Ä
/generator/G_MODEL/E/LayerNorm_1/batchnorm/mul_2Mul,generator/G_MODEL/E/LayerNorm_1/moments/mean-generator/G_MODEL/E/LayerNorm_1/batchnorm/mul*
T0*&
_output_shapes
:@
Á
-generator/G_MODEL/E/LayerNorm_1/batchnorm/subSub)generator/G_MODEL/E/LayerNorm_1/beta/read/generator/G_MODEL/E/LayerNorm_1/batchnorm/mul_2*
T0*&
_output_shapes
:@
Û
/generator/G_MODEL/E/LayerNorm_1/batchnorm/add_1AddV2/generator/G_MODEL/E/LayerNorm_1/batchnorm/mul_1-generator/G_MODEL/E/LayerNorm_1/batchnorm/sub*
T0*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@
°
generator/G_MODEL/E/LeakyRelu_1	LeakyRelu/generator/G_MODEL/E/LayerNorm_1/batchnorm/add_1*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
T0*
alpha%ÍĖL>

(generator/G_MODEL/E/MirrorPad_2/paddingsConst*
dtype0*9
value0B."                             *
_output_shapes

:
Þ
generator/G_MODEL/E/MirrorPad_2	MirrorPadgenerator/G_MODEL/E/LeakyRelu_1(generator/G_MODEL/E/MirrorPad_2/paddings*
mode	REFLECT*
T0*
	Tpaddings0*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@
Õ
Egenerator/G_MODEL/E/Conv_2/weights/Initializer/truncated_normal/shapeConst*5
_class+
)'loc:@generator/G_MODEL/E/Conv_2/weights*
_output_shapes
:*
dtype0*%
valueB"      @       
Ā
Dgenerator/G_MODEL/E/Conv_2/weights/Initializer/truncated_normal/meanConst*5
_class+
)'loc:@generator/G_MODEL/E/Conv_2/weights*
_output_shapes
: *
valueB
 *    *
dtype0
Â
Fgenerator/G_MODEL/E/Conv_2/weights/Initializer/truncated_normal/stddevConst*
valueB
 *îāë<*
_output_shapes
: *
dtype0*5
_class+
)'loc:@generator/G_MODEL/E/Conv_2/weights
·
Ogenerator/G_MODEL/E/Conv_2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalEgenerator/G_MODEL/E/Conv_2/weights/Initializer/truncated_normal/shape*

seed *&
_output_shapes
:@ *
T0*
seed2 *
dtype0*5
_class+
)'loc:@generator/G_MODEL/E/Conv_2/weights
Ë
Cgenerator/G_MODEL/E/Conv_2/weights/Initializer/truncated_normal/mulMulOgenerator/G_MODEL/E/Conv_2/weights/Initializer/truncated_normal/TruncatedNormalFgenerator/G_MODEL/E/Conv_2/weights/Initializer/truncated_normal/stddev*
T0*5
_class+
)'loc:@generator/G_MODEL/E/Conv_2/weights*&
_output_shapes
:@ 
đ
?generator/G_MODEL/E/Conv_2/weights/Initializer/truncated_normalAddCgenerator/G_MODEL/E/Conv_2/weights/Initializer/truncated_normal/mulDgenerator/G_MODEL/E/Conv_2/weights/Initializer/truncated_normal/mean*5
_class+
)'loc:@generator/G_MODEL/E/Conv_2/weights*&
_output_shapes
:@ *
T0
Ý
"generator/G_MODEL/E/Conv_2/weights
VariableV2*
shared_name *5
_class+
)'loc:@generator/G_MODEL/E/Conv_2/weights*&
_output_shapes
:@ *
dtype0*
	container *
shape:@ 
Đ
)generator/G_MODEL/E/Conv_2/weights/AssignAssign"generator/G_MODEL/E/Conv_2/weights?generator/G_MODEL/E/Conv_2/weights/Initializer/truncated_normal*
use_locking(*
T0*
validate_shape(*&
_output_shapes
:@ *5
_class+
)'loc:@generator/G_MODEL/E/Conv_2/weights
ŋ
'generator/G_MODEL/E/Conv_2/weights/readIdentity"generator/G_MODEL/E/Conv_2/weights*5
_class+
)'loc:@generator/G_MODEL/E/Conv_2/weights*&
_output_shapes
:@ *
T0
y
(generator/G_MODEL/E/Conv_2/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ā
!generator/G_MODEL/E/Conv_2/Conv2DConv2Dgenerator/G_MODEL/E/MirrorPad_2'generator/G_MODEL/E/Conv_2/weights/read*
explicit_paddings
 *
strides
*
	dilations
*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ *
data_formatNHWC*
T0*
use_cudnn_on_gpu(*
paddingVALID
ž
6generator/G_MODEL/E/LayerNorm_2/beta/Initializer/zerosConst*
_output_shapes
: *
valueB *    *7
_class-
+)loc:@generator/G_MODEL/E/LayerNorm_2/beta*
dtype0
É
$generator/G_MODEL/E/LayerNorm_2/beta
VariableV2*
shared_name *
shape: *7
_class-
+)loc:@generator/G_MODEL/E/LayerNorm_2/beta*
	container *
dtype0*
_output_shapes
: 

+generator/G_MODEL/E/LayerNorm_2/beta/AssignAssign$generator/G_MODEL/E/LayerNorm_2/beta6generator/G_MODEL/E/LayerNorm_2/beta/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@generator/G_MODEL/E/LayerNorm_2/beta*
_output_shapes
: 
đ
)generator/G_MODEL/E/LayerNorm_2/beta/readIdentity$generator/G_MODEL/E/LayerNorm_2/beta*
_output_shapes
: *7
_class-
+)loc:@generator/G_MODEL/E/LayerNorm_2/beta*
T0
―
6generator/G_MODEL/E/LayerNorm_2/gamma/Initializer/onesConst*
dtype0*
valueB *  ?*8
_class.
,*loc:@generator/G_MODEL/E/LayerNorm_2/gamma*
_output_shapes
: 
Ë
%generator/G_MODEL/E/LayerNorm_2/gamma
VariableV2*
	container *
shape: *
_output_shapes
: *
dtype0*
shared_name *8
_class.
,*loc:@generator/G_MODEL/E/LayerNorm_2/gamma

,generator/G_MODEL/E/LayerNorm_2/gamma/AssignAssign%generator/G_MODEL/E/LayerNorm_2/gamma6generator/G_MODEL/E/LayerNorm_2/gamma/Initializer/ones*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@generator/G_MODEL/E/LayerNorm_2/gamma*
_output_shapes
: 
ž
*generator/G_MODEL/E/LayerNorm_2/gamma/readIdentity%generator/G_MODEL/E/LayerNorm_2/gamma*8
_class.
,*loc:@generator/G_MODEL/E/LayerNorm_2/gamma*
T0*
_output_shapes
: 

>generator/G_MODEL/E/LayerNorm_2/moments/mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:
å
,generator/G_MODEL/E/LayerNorm_2/moments/meanMean!generator/G_MODEL/E/Conv_2/Conv2D>generator/G_MODEL/E/LayerNorm_2/moments/mean/reduction_indices*

Tidx0*&
_output_shapes
:*
	keep_dims(*
T0
Ģ
4generator/G_MODEL/E/LayerNorm_2/moments/StopGradientStopGradient,generator/G_MODEL/E/LayerNorm_2/moments/mean*
T0*&
_output_shapes
:
ę
9generator/G_MODEL/E/LayerNorm_2/moments/SquaredDifferenceSquaredDifference!generator/G_MODEL/E/Conv_2/Conv2D4generator/G_MODEL/E/LayerNorm_2/moments/StopGradient*
T0*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ 

Bgenerator/G_MODEL/E/LayerNorm_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         

0generator/G_MODEL/E/LayerNorm_2/moments/varianceMean9generator/G_MODEL/E/LayerNorm_2/moments/SquaredDifferenceBgenerator/G_MODEL/E/LayerNorm_2/moments/variance/reduction_indices*

Tidx0*&
_output_shapes
:*
T0*
	keep_dims(
t
/generator/G_MODEL/E/LayerNorm_2/batchnorm/add/yConst*
valueB
 *Ėž+*
_output_shapes
: *
dtype0
Ę
-generator/G_MODEL/E/LayerNorm_2/batchnorm/addAddV20generator/G_MODEL/E/LayerNorm_2/moments/variance/generator/G_MODEL/E/LayerNorm_2/batchnorm/add/y*&
_output_shapes
:*
T0

/generator/G_MODEL/E/LayerNorm_2/batchnorm/RsqrtRsqrt-generator/G_MODEL/E/LayerNorm_2/batchnorm/add*&
_output_shapes
:*
T0
Â
-generator/G_MODEL/E/LayerNorm_2/batchnorm/mulMul/generator/G_MODEL/E/LayerNorm_2/batchnorm/Rsqrt*generator/G_MODEL/E/LayerNorm_2/gamma/read*
T0*&
_output_shapes
: 
Ë
/generator/G_MODEL/E/LayerNorm_2/batchnorm/mul_1Mul!generator/G_MODEL/E/Conv_2/Conv2D-generator/G_MODEL/E/LayerNorm_2/batchnorm/mul*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ *
T0
Ä
/generator/G_MODEL/E/LayerNorm_2/batchnorm/mul_2Mul,generator/G_MODEL/E/LayerNorm_2/moments/mean-generator/G_MODEL/E/LayerNorm_2/batchnorm/mul*&
_output_shapes
: *
T0
Á
-generator/G_MODEL/E/LayerNorm_2/batchnorm/subSub)generator/G_MODEL/E/LayerNorm_2/beta/read/generator/G_MODEL/E/LayerNorm_2/batchnorm/mul_2*
T0*&
_output_shapes
: 
Û
/generator/G_MODEL/E/LayerNorm_2/batchnorm/add_1AddV2/generator/G_MODEL/E/LayerNorm_2/batchnorm/mul_1-generator/G_MODEL/E/LayerNorm_2/batchnorm/sub*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ *
T0
°
generator/G_MODEL/E/LeakyRelu_2	LeakyRelu/generator/G_MODEL/E/LayerNorm_2/batchnorm/add_1*
T0*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ *
alpha%ÍĖL>
á
Kgenerator/G_MODEL/out_layer/Conv/weights/Initializer/truncated_normal/shapeConst*;
_class1
/-loc:@generator/G_MODEL/out_layer/Conv/weights*
dtype0*
_output_shapes
:*%
valueB"             
Ė
Jgenerator/G_MODEL/out_layer/Conv/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *;
_class1
/-loc:@generator/G_MODEL/out_layer/Conv/weights*
dtype0*
_output_shapes
: 
Î
Lgenerator/G_MODEL/out_layer/Conv/weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *Eņ>*;
_class1
/-loc:@generator/G_MODEL/out_layer/Conv/weights*
dtype0
É
Ugenerator/G_MODEL/out_layer/Conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKgenerator/G_MODEL/out_layer/Conv/weights/Initializer/truncated_normal/shape*;
_class1
/-loc:@generator/G_MODEL/out_layer/Conv/weights*

seed *
dtype0*
seed2 *&
_output_shapes
: *
T0
ã
Igenerator/G_MODEL/out_layer/Conv/weights/Initializer/truncated_normal/mulMulUgenerator/G_MODEL/out_layer/Conv/weights/Initializer/truncated_normal/TruncatedNormalLgenerator/G_MODEL/out_layer/Conv/weights/Initializer/truncated_normal/stddev*;
_class1
/-loc:@generator/G_MODEL/out_layer/Conv/weights*
T0*&
_output_shapes
: 
Ņ
Egenerator/G_MODEL/out_layer/Conv/weights/Initializer/truncated_normalAddIgenerator/G_MODEL/out_layer/Conv/weights/Initializer/truncated_normal/mulJgenerator/G_MODEL/out_layer/Conv/weights/Initializer/truncated_normal/mean*
T0*;
_class1
/-loc:@generator/G_MODEL/out_layer/Conv/weights*&
_output_shapes
: 
é
(generator/G_MODEL/out_layer/Conv/weights
VariableV2*
shape: *
shared_name *
	container *
dtype0*&
_output_shapes
: *;
_class1
/-loc:@generator/G_MODEL/out_layer/Conv/weights
Á
/generator/G_MODEL/out_layer/Conv/weights/AssignAssign(generator/G_MODEL/out_layer/Conv/weightsEgenerator/G_MODEL/out_layer/Conv/weights/Initializer/truncated_normal*;
_class1
/-loc:@generator/G_MODEL/out_layer/Conv/weights*
validate_shape(*
use_locking(*
T0*&
_output_shapes
: 
Ņ
-generator/G_MODEL/out_layer/Conv/weights/readIdentity(generator/G_MODEL/out_layer/Conv/weights*;
_class1
/-loc:@generator/G_MODEL/out_layer/Conv/weights*
T0*&
_output_shapes
: 

.generator/G_MODEL/out_layer/Conv/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
Ė
'generator/G_MODEL/out_layer/Conv/Conv2DConv2Dgenerator/G_MODEL/E/LeakyRelu_2-generator/G_MODEL/out_layer/Conv/weights/read*
strides
*
data_formatNHWC*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
explicit_paddings
 *
T0*
	dilations
*
paddingVALID*
use_cudnn_on_gpu(

 generator/G_MODEL/out_layer/TanhTanh'generator/G_MODEL/out_layer/Conv/Conv2D*8
_output_shapes&
$:"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
T0
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
_output_shapes
: *
dtype0

save/SaveV2/tensor_namesConst*
dtype0*Å
valueŧBļMB generator/G_MODEL/A/Conv/weightsB"generator/G_MODEL/A/Conv_1/weightsB"generator/G_MODEL/A/Conv_2/weightsB"generator/G_MODEL/A/LayerNorm/betaB#generator/G_MODEL/A/LayerNorm/gammaB$generator/G_MODEL/A/LayerNorm_1/betaB%generator/G_MODEL/A/LayerNorm_1/gammaB$generator/G_MODEL/A/LayerNorm_2/betaB%generator/G_MODEL/A/LayerNorm_2/gammaB generator/G_MODEL/B/Conv/weightsB"generator/G_MODEL/B/Conv_1/weightsB"generator/G_MODEL/B/LayerNorm/betaB#generator/G_MODEL/B/LayerNorm/gammaB$generator/G_MODEL/B/LayerNorm_1/betaB%generator/G_MODEL/B/LayerNorm_1/gammaB generator/G_MODEL/C/Conv/weightsB"generator/G_MODEL/C/Conv_1/weightsB"generator/G_MODEL/C/LayerNorm/betaB#generator/G_MODEL/C/LayerNorm/gammaB$generator/G_MODEL/C/LayerNorm_1/betaB%generator/G_MODEL/C/LayerNorm_1/gammaBgenerator/G_MODEL/C/r1/1/betaBgenerator/G_MODEL/C/r1/1/gammaBgenerator/G_MODEL/C/r1/2/betaBgenerator/G_MODEL/C/r1/2/gammaB#generator/G_MODEL/C/r1/Conv/weightsB%generator/G_MODEL/C/r1/Conv_1/weightsB%generator/G_MODEL/C/r1/LayerNorm/betaB&generator/G_MODEL/C/r1/LayerNorm/gammaBgenerator/G_MODEL/C/r1/r1/biasBgenerator/G_MODEL/C/r1/r1/wBgenerator/G_MODEL/C/r2/1/betaBgenerator/G_MODEL/C/r2/1/gammaBgenerator/G_MODEL/C/r2/2/betaBgenerator/G_MODEL/C/r2/2/gammaB#generator/G_MODEL/C/r2/Conv/weightsB%generator/G_MODEL/C/r2/Conv_1/weightsB%generator/G_MODEL/C/r2/LayerNorm/betaB&generator/G_MODEL/C/r2/LayerNorm/gammaBgenerator/G_MODEL/C/r2/r2/biasBgenerator/G_MODEL/C/r2/r2/wBgenerator/G_MODEL/C/r3/1/betaBgenerator/G_MODEL/C/r3/1/gammaBgenerator/G_MODEL/C/r3/2/betaBgenerator/G_MODEL/C/r3/2/gammaB#generator/G_MODEL/C/r3/Conv/weightsB%generator/G_MODEL/C/r3/Conv_1/weightsB%generator/G_MODEL/C/r3/LayerNorm/betaB&generator/G_MODEL/C/r3/LayerNorm/gammaBgenerator/G_MODEL/C/r3/r3/biasBgenerator/G_MODEL/C/r3/r3/wBgenerator/G_MODEL/C/r4/1/betaBgenerator/G_MODEL/C/r4/1/gammaBgenerator/G_MODEL/C/r4/2/betaBgenerator/G_MODEL/C/r4/2/gammaB#generator/G_MODEL/C/r4/Conv/weightsB%generator/G_MODEL/C/r4/Conv_1/weightsB%generator/G_MODEL/C/r4/LayerNorm/betaB&generator/G_MODEL/C/r4/LayerNorm/gammaBgenerator/G_MODEL/C/r4/r4/biasBgenerator/G_MODEL/C/r4/r4/wB generator/G_MODEL/D/Conv/weightsB"generator/G_MODEL/D/Conv_1/weightsB"generator/G_MODEL/D/LayerNorm/betaB#generator/G_MODEL/D/LayerNorm/gammaB$generator/G_MODEL/D/LayerNorm_1/betaB%generator/G_MODEL/D/LayerNorm_1/gammaB generator/G_MODEL/E/Conv/weightsB"generator/G_MODEL/E/Conv_1/weightsB"generator/G_MODEL/E/Conv_2/weightsB"generator/G_MODEL/E/LayerNorm/betaB#generator/G_MODEL/E/LayerNorm/gammaB$generator/G_MODEL/E/LayerNorm_1/betaB%generator/G_MODEL/E/LayerNorm_1/gammaB$generator/G_MODEL/E/LayerNorm_2/betaB%generator/G_MODEL/E/LayerNorm_2/gammaB(generator/G_MODEL/out_layer/Conv/weights*
_output_shapes
:M

save/SaveV2/shape_and_slicesConst*
_output_shapes
:M*
dtype0*Ŋ
valueĨBĒMB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
æ
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices generator/G_MODEL/A/Conv/weights"generator/G_MODEL/A/Conv_1/weights"generator/G_MODEL/A/Conv_2/weights"generator/G_MODEL/A/LayerNorm/beta#generator/G_MODEL/A/LayerNorm/gamma$generator/G_MODEL/A/LayerNorm_1/beta%generator/G_MODEL/A/LayerNorm_1/gamma$generator/G_MODEL/A/LayerNorm_2/beta%generator/G_MODEL/A/LayerNorm_2/gamma generator/G_MODEL/B/Conv/weights"generator/G_MODEL/B/Conv_1/weights"generator/G_MODEL/B/LayerNorm/beta#generator/G_MODEL/B/LayerNorm/gamma$generator/G_MODEL/B/LayerNorm_1/beta%generator/G_MODEL/B/LayerNorm_1/gamma generator/G_MODEL/C/Conv/weights"generator/G_MODEL/C/Conv_1/weights"generator/G_MODEL/C/LayerNorm/beta#generator/G_MODEL/C/LayerNorm/gamma$generator/G_MODEL/C/LayerNorm_1/beta%generator/G_MODEL/C/LayerNorm_1/gammagenerator/G_MODEL/C/r1/1/betagenerator/G_MODEL/C/r1/1/gammagenerator/G_MODEL/C/r1/2/betagenerator/G_MODEL/C/r1/2/gamma#generator/G_MODEL/C/r1/Conv/weights%generator/G_MODEL/C/r1/Conv_1/weights%generator/G_MODEL/C/r1/LayerNorm/beta&generator/G_MODEL/C/r1/LayerNorm/gammagenerator/G_MODEL/C/r1/r1/biasgenerator/G_MODEL/C/r1/r1/wgenerator/G_MODEL/C/r2/1/betagenerator/G_MODEL/C/r2/1/gammagenerator/G_MODEL/C/r2/2/betagenerator/G_MODEL/C/r2/2/gamma#generator/G_MODEL/C/r2/Conv/weights%generator/G_MODEL/C/r2/Conv_1/weights%generator/G_MODEL/C/r2/LayerNorm/beta&generator/G_MODEL/C/r2/LayerNorm/gammagenerator/G_MODEL/C/r2/r2/biasgenerator/G_MODEL/C/r2/r2/wgenerator/G_MODEL/C/r3/1/betagenerator/G_MODEL/C/r3/1/gammagenerator/G_MODEL/C/r3/2/betagenerator/G_MODEL/C/r3/2/gamma#generator/G_MODEL/C/r3/Conv/weights%generator/G_MODEL/C/r3/Conv_1/weights%generator/G_MODEL/C/r3/LayerNorm/beta&generator/G_MODEL/C/r3/LayerNorm/gammagenerator/G_MODEL/C/r3/r3/biasgenerator/G_MODEL/C/r3/r3/wgenerator/G_MODEL/C/r4/1/betagenerator/G_MODEL/C/r4/1/gammagenerator/G_MODEL/C/r4/2/betagenerator/G_MODEL/C/r4/2/gamma#generator/G_MODEL/C/r4/Conv/weights%generator/G_MODEL/C/r4/Conv_1/weights%generator/G_MODEL/C/r4/LayerNorm/beta&generator/G_MODEL/C/r4/LayerNorm/gammagenerator/G_MODEL/C/r4/r4/biasgenerator/G_MODEL/C/r4/r4/w generator/G_MODEL/D/Conv/weights"generator/G_MODEL/D/Conv_1/weights"generator/G_MODEL/D/LayerNorm/beta#generator/G_MODEL/D/LayerNorm/gamma$generator/G_MODEL/D/LayerNorm_1/beta%generator/G_MODEL/D/LayerNorm_1/gamma generator/G_MODEL/E/Conv/weights"generator/G_MODEL/E/Conv_1/weights"generator/G_MODEL/E/Conv_2/weights"generator/G_MODEL/E/LayerNorm/beta#generator/G_MODEL/E/LayerNorm/gamma$generator/G_MODEL/E/LayerNorm_1/beta%generator/G_MODEL/E/LayerNorm_1/gamma$generator/G_MODEL/E/LayerNorm_2/beta%generator/G_MODEL/E/LayerNorm_2/gamma(generator/G_MODEL/out_layer/Conv/weights*[
dtypesQ
O2M
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
Ī
save/RestoreV2/tensor_namesConst"/device:CPU:0*Å
valueŧBļMB generator/G_MODEL/A/Conv/weightsB"generator/G_MODEL/A/Conv_1/weightsB"generator/G_MODEL/A/Conv_2/weightsB"generator/G_MODEL/A/LayerNorm/betaB#generator/G_MODEL/A/LayerNorm/gammaB$generator/G_MODEL/A/LayerNorm_1/betaB%generator/G_MODEL/A/LayerNorm_1/gammaB$generator/G_MODEL/A/LayerNorm_2/betaB%generator/G_MODEL/A/LayerNorm_2/gammaB generator/G_MODEL/B/Conv/weightsB"generator/G_MODEL/B/Conv_1/weightsB"generator/G_MODEL/B/LayerNorm/betaB#generator/G_MODEL/B/LayerNorm/gammaB$generator/G_MODEL/B/LayerNorm_1/betaB%generator/G_MODEL/B/LayerNorm_1/gammaB generator/G_MODEL/C/Conv/weightsB"generator/G_MODEL/C/Conv_1/weightsB"generator/G_MODEL/C/LayerNorm/betaB#generator/G_MODEL/C/LayerNorm/gammaB$generator/G_MODEL/C/LayerNorm_1/betaB%generator/G_MODEL/C/LayerNorm_1/gammaBgenerator/G_MODEL/C/r1/1/betaBgenerator/G_MODEL/C/r1/1/gammaBgenerator/G_MODEL/C/r1/2/betaBgenerator/G_MODEL/C/r1/2/gammaB#generator/G_MODEL/C/r1/Conv/weightsB%generator/G_MODEL/C/r1/Conv_1/weightsB%generator/G_MODEL/C/r1/LayerNorm/betaB&generator/G_MODEL/C/r1/LayerNorm/gammaBgenerator/G_MODEL/C/r1/r1/biasBgenerator/G_MODEL/C/r1/r1/wBgenerator/G_MODEL/C/r2/1/betaBgenerator/G_MODEL/C/r2/1/gammaBgenerator/G_MODEL/C/r2/2/betaBgenerator/G_MODEL/C/r2/2/gammaB#generator/G_MODEL/C/r2/Conv/weightsB%generator/G_MODEL/C/r2/Conv_1/weightsB%generator/G_MODEL/C/r2/LayerNorm/betaB&generator/G_MODEL/C/r2/LayerNorm/gammaBgenerator/G_MODEL/C/r2/r2/biasBgenerator/G_MODEL/C/r2/r2/wBgenerator/G_MODEL/C/r3/1/betaBgenerator/G_MODEL/C/r3/1/gammaBgenerator/G_MODEL/C/r3/2/betaBgenerator/G_MODEL/C/r3/2/gammaB#generator/G_MODEL/C/r3/Conv/weightsB%generator/G_MODEL/C/r3/Conv_1/weightsB%generator/G_MODEL/C/r3/LayerNorm/betaB&generator/G_MODEL/C/r3/LayerNorm/gammaBgenerator/G_MODEL/C/r3/r3/biasBgenerator/G_MODEL/C/r3/r3/wBgenerator/G_MODEL/C/r4/1/betaBgenerator/G_MODEL/C/r4/1/gammaBgenerator/G_MODEL/C/r4/2/betaBgenerator/G_MODEL/C/r4/2/gammaB#generator/G_MODEL/C/r4/Conv/weightsB%generator/G_MODEL/C/r4/Conv_1/weightsB%generator/G_MODEL/C/r4/LayerNorm/betaB&generator/G_MODEL/C/r4/LayerNorm/gammaBgenerator/G_MODEL/C/r4/r4/biasBgenerator/G_MODEL/C/r4/r4/wB generator/G_MODEL/D/Conv/weightsB"generator/G_MODEL/D/Conv_1/weightsB"generator/G_MODEL/D/LayerNorm/betaB#generator/G_MODEL/D/LayerNorm/gammaB$generator/G_MODEL/D/LayerNorm_1/betaB%generator/G_MODEL/D/LayerNorm_1/gammaB generator/G_MODEL/E/Conv/weightsB"generator/G_MODEL/E/Conv_1/weightsB"generator/G_MODEL/E/Conv_2/weightsB"generator/G_MODEL/E/LayerNorm/betaB#generator/G_MODEL/E/LayerNorm/gammaB$generator/G_MODEL/E/LayerNorm_1/betaB%generator/G_MODEL/E/LayerNorm_1/gammaB$generator/G_MODEL/E/LayerNorm_2/betaB%generator/G_MODEL/E/LayerNorm_2/gammaB(generator/G_MODEL/out_layer/Conv/weights*
_output_shapes
:M*
dtype0

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*Ŋ
valueĨBĒMB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:M

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*Ę
_output_shapes·
ī:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*[
dtypesQ
O2M
Ö
save/AssignAssign generator/G_MODEL/A/Conv/weightssave/RestoreV2*3
_class)
'%loc:@generator/G_MODEL/A/Conv/weights*&
_output_shapes
: *
validate_shape(*
T0*
use_locking(
Þ
save/Assign_1Assign"generator/G_MODEL/A/Conv_1/weightssave/RestoreV2:1*
T0*&
_output_shapes
: @*
validate_shape(*
use_locking(*5
_class+
)'loc:@generator/G_MODEL/A/Conv_1/weights
Þ
save/Assign_2Assign"generator/G_MODEL/A/Conv_2/weightssave/RestoreV2:2*&
_output_shapes
:@@*
use_locking(*
T0*
validate_shape(*5
_class+
)'loc:@generator/G_MODEL/A/Conv_2/weights
Ō
save/Assign_3Assign"generator/G_MODEL/A/LayerNorm/betasave/RestoreV2:3*
_output_shapes
: *
validate_shape(*
use_locking(*5
_class+
)'loc:@generator/G_MODEL/A/LayerNorm/beta*
T0
Ô
save/Assign_4Assign#generator/G_MODEL/A/LayerNorm/gammasave/RestoreV2:4*
T0*
use_locking(*
_output_shapes
: *6
_class,
*(loc:@generator/G_MODEL/A/LayerNorm/gamma*
validate_shape(
Ö
save/Assign_5Assign$generator/G_MODEL/A/LayerNorm_1/betasave/RestoreV2:5*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@*7
_class-
+)loc:@generator/G_MODEL/A/LayerNorm_1/beta
Ø
save/Assign_6Assign%generator/G_MODEL/A/LayerNorm_1/gammasave/RestoreV2:6*8
_class.
,*loc:@generator/G_MODEL/A/LayerNorm_1/gamma*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(
Ö
save/Assign_7Assign$generator/G_MODEL/A/LayerNorm_2/betasave/RestoreV2:7*
use_locking(*
validate_shape(*7
_class-
+)loc:@generator/G_MODEL/A/LayerNorm_2/beta*
_output_shapes
:@*
T0
Ø
save/Assign_8Assign%generator/G_MODEL/A/LayerNorm_2/gammasave/RestoreV2:8*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@*8
_class.
,*loc:@generator/G_MODEL/A/LayerNorm_2/gamma
Û
save/Assign_9Assign generator/G_MODEL/B/Conv/weightssave/RestoreV2:9*
validate_shape(*
T0*'
_output_shapes
:@*3
_class)
'%loc:@generator/G_MODEL/B/Conv/weights*
use_locking(
â
save/Assign_10Assign"generator/G_MODEL/B/Conv_1/weightssave/RestoreV2:10*
T0*(
_output_shapes
:*
validate_shape(*
use_locking(*5
_class+
)'loc:@generator/G_MODEL/B/Conv_1/weights
Õ
save/Assign_11Assign"generator/G_MODEL/B/LayerNorm/betasave/RestoreV2:11*
T0*5
_class+
)'loc:@generator/G_MODEL/B/LayerNorm/beta*
_output_shapes	
:*
validate_shape(*
use_locking(
Ũ
save/Assign_12Assign#generator/G_MODEL/B/LayerNorm/gammasave/RestoreV2:12*
use_locking(*
validate_shape(*
T0*6
_class,
*(loc:@generator/G_MODEL/B/LayerNorm/gamma*
_output_shapes	
:
Ų
save/Assign_13Assign$generator/G_MODEL/B/LayerNorm_1/betasave/RestoreV2:13*
T0*7
_class-
+)loc:@generator/G_MODEL/B/LayerNorm_1/beta*
validate_shape(*
use_locking(*
_output_shapes	
:
Û
save/Assign_14Assign%generator/G_MODEL/B/LayerNorm_1/gammasave/RestoreV2:14*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0*8
_class.
,*loc:@generator/G_MODEL/B/LayerNorm_1/gamma
Þ
save/Assign_15Assign generator/G_MODEL/C/Conv/weightssave/RestoreV2:15*
validate_shape(*
T0*(
_output_shapes
:*
use_locking(*3
_class)
'%loc:@generator/G_MODEL/C/Conv/weights
â
save/Assign_16Assign"generator/G_MODEL/C/Conv_1/weightssave/RestoreV2:16*(
_output_shapes
:*
use_locking(*5
_class+
)'loc:@generator/G_MODEL/C/Conv_1/weights*
validate_shape(*
T0
Õ
save/Assign_17Assign"generator/G_MODEL/C/LayerNorm/betasave/RestoreV2:17*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:*5
_class+
)'loc:@generator/G_MODEL/C/LayerNorm/beta
Ũ
save/Assign_18Assign#generator/G_MODEL/C/LayerNorm/gammasave/RestoreV2:18*
_output_shapes	
:*
T0*
validate_shape(*6
_class,
*(loc:@generator/G_MODEL/C/LayerNorm/gamma*
use_locking(
Ų
save/Assign_19Assign$generator/G_MODEL/C/LayerNorm_1/betasave/RestoreV2:19*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0*7
_class-
+)loc:@generator/G_MODEL/C/LayerNorm_1/beta
Û
save/Assign_20Assign%generator/G_MODEL/C/LayerNorm_1/gammasave/RestoreV2:20*8
_class.
,*loc:@generator/G_MODEL/C/LayerNorm_1/gamma*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0
Ë
save/Assign_21Assigngenerator/G_MODEL/C/r1/1/betasave/RestoreV2:21*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@generator/G_MODEL/C/r1/1/beta
Í
save/Assign_22Assigngenerator/G_MODEL/C/r1/1/gammasave/RestoreV2:22*
use_locking(*1
_class'
%#loc:@generator/G_MODEL/C/r1/1/gamma*
validate_shape(*
_output_shapes	
:*
T0
Ë
save/Assign_23Assigngenerator/G_MODEL/C/r1/2/betasave/RestoreV2:23*
T0*
_output_shapes	
:*
use_locking(*0
_class&
$"loc:@generator/G_MODEL/C/r1/2/beta*
validate_shape(
Í
save/Assign_24Assigngenerator/G_MODEL/C/r1/2/gammasave/RestoreV2:24*
T0*
validate_shape(*1
_class'
%#loc:@generator/G_MODEL/C/r1/2/gamma*
use_locking(*
_output_shapes	
:
ä
save/Assign_25Assign#generator/G_MODEL/C/r1/Conv/weightssave/RestoreV2:25*6
_class,
*(loc:@generator/G_MODEL/C/r1/Conv/weights*
validate_shape(*
T0*(
_output_shapes
:*
use_locking(
č
save/Assign_26Assign%generator/G_MODEL/C/r1/Conv_1/weightssave/RestoreV2:26*
use_locking(*
validate_shape(*(
_output_shapes
:*8
_class.
,*loc:@generator/G_MODEL/C/r1/Conv_1/weights*
T0
Û
save/Assign_27Assign%generator/G_MODEL/C/r1/LayerNorm/betasave/RestoreV2:27*
T0*
use_locking(*
_output_shapes	
:*8
_class.
,*loc:@generator/G_MODEL/C/r1/LayerNorm/beta*
validate_shape(
Ý
save/Assign_28Assign&generator/G_MODEL/C/r1/LayerNorm/gammasave/RestoreV2:28*9
_class/
-+loc:@generator/G_MODEL/C/r1/LayerNorm/gamma*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
Í
save/Assign_29Assigngenerator/G_MODEL/C/r1/r1/biassave/RestoreV2:29*
T0*
_output_shapes	
:*1
_class'
%#loc:@generator/G_MODEL/C/r1/r1/bias*
validate_shape(*
use_locking(
Ó
save/Assign_30Assigngenerator/G_MODEL/C/r1/r1/wsave/RestoreV2:30*
use_locking(*
validate_shape(*
T0*.
_class$
" loc:@generator/G_MODEL/C/r1/r1/w*'
_output_shapes
:
Ë
save/Assign_31Assigngenerator/G_MODEL/C/r2/1/betasave/RestoreV2:31*0
_class&
$"loc:@generator/G_MODEL/C/r2/1/beta*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:
Í
save/Assign_32Assigngenerator/G_MODEL/C/r2/1/gammasave/RestoreV2:32*
use_locking(*
_output_shapes	
:*1
_class'
%#loc:@generator/G_MODEL/C/r2/1/gamma*
T0*
validate_shape(
Ë
save/Assign_33Assigngenerator/G_MODEL/C/r2/2/betasave/RestoreV2:33*0
_class&
$"loc:@generator/G_MODEL/C/r2/2/beta*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:
Í
save/Assign_34Assigngenerator/G_MODEL/C/r2/2/gammasave/RestoreV2:34*1
_class'
%#loc:@generator/G_MODEL/C/r2/2/gamma*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
ä
save/Assign_35Assign#generator/G_MODEL/C/r2/Conv/weightssave/RestoreV2:35*
T0*6
_class,
*(loc:@generator/G_MODEL/C/r2/Conv/weights*(
_output_shapes
:*
validate_shape(*
use_locking(
č
save/Assign_36Assign%generator/G_MODEL/C/r2/Conv_1/weightssave/RestoreV2:36*
T0*
validate_shape(*(
_output_shapes
:*8
_class.
,*loc:@generator/G_MODEL/C/r2/Conv_1/weights*
use_locking(
Û
save/Assign_37Assign%generator/G_MODEL/C/r2/LayerNorm/betasave/RestoreV2:37*
_output_shapes	
:*8
_class.
,*loc:@generator/G_MODEL/C/r2/LayerNorm/beta*
validate_shape(*
T0*
use_locking(
Ý
save/Assign_38Assign&generator/G_MODEL/C/r2/LayerNorm/gammasave/RestoreV2:38*
T0*
_output_shapes	
:*
validate_shape(*9
_class/
-+loc:@generator/G_MODEL/C/r2/LayerNorm/gamma*
use_locking(
Í
save/Assign_39Assigngenerator/G_MODEL/C/r2/r2/biassave/RestoreV2:39*
use_locking(*1
_class'
%#loc:@generator/G_MODEL/C/r2/r2/bias*
validate_shape(*
_output_shapes	
:*
T0
Ó
save/Assign_40Assigngenerator/G_MODEL/C/r2/r2/wsave/RestoreV2:40*'
_output_shapes
:*.
_class$
" loc:@generator/G_MODEL/C/r2/r2/w*
validate_shape(*
T0*
use_locking(
Ë
save/Assign_41Assigngenerator/G_MODEL/C/r3/1/betasave/RestoreV2:41*
validate_shape(*
use_locking(*
_output_shapes	
:*0
_class&
$"loc:@generator/G_MODEL/C/r3/1/beta*
T0
Í
save/Assign_42Assigngenerator/G_MODEL/C/r3/1/gammasave/RestoreV2:42*
_output_shapes	
:*
T0*1
_class'
%#loc:@generator/G_MODEL/C/r3/1/gamma*
validate_shape(*
use_locking(
Ë
save/Assign_43Assigngenerator/G_MODEL/C/r3/2/betasave/RestoreV2:43*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(*0
_class&
$"loc:@generator/G_MODEL/C/r3/2/beta
Í
save/Assign_44Assigngenerator/G_MODEL/C/r3/2/gammasave/RestoreV2:44*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(*1
_class'
%#loc:@generator/G_MODEL/C/r3/2/gamma
ä
save/Assign_45Assign#generator/G_MODEL/C/r3/Conv/weightssave/RestoreV2:45*
T0*6
_class,
*(loc:@generator/G_MODEL/C/r3/Conv/weights*
use_locking(*
validate_shape(*(
_output_shapes
:
č
save/Assign_46Assign%generator/G_MODEL/C/r3/Conv_1/weightssave/RestoreV2:46*
T0*(
_output_shapes
:*
validate_shape(*8
_class.
,*loc:@generator/G_MODEL/C/r3/Conv_1/weights*
use_locking(
Û
save/Assign_47Assign%generator/G_MODEL/C/r3/LayerNorm/betasave/RestoreV2:47*8
_class.
,*loc:@generator/G_MODEL/C/r3/LayerNorm/beta*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(
Ý
save/Assign_48Assign&generator/G_MODEL/C/r3/LayerNorm/gammasave/RestoreV2:48*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:*9
_class/
-+loc:@generator/G_MODEL/C/r3/LayerNorm/gamma
Í
save/Assign_49Assigngenerator/G_MODEL/C/r3/r3/biassave/RestoreV2:49*
validate_shape(*1
_class'
%#loc:@generator/G_MODEL/C/r3/r3/bias*
_output_shapes	
:*
use_locking(*
T0
Ó
save/Assign_50Assigngenerator/G_MODEL/C/r3/r3/wsave/RestoreV2:50*.
_class$
" loc:@generator/G_MODEL/C/r3/r3/w*'
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Ë
save/Assign_51Assigngenerator/G_MODEL/C/r4/1/betasave/RestoreV2:51*
T0*
_output_shapes	
:*0
_class&
$"loc:@generator/G_MODEL/C/r4/1/beta*
use_locking(*
validate_shape(
Í
save/Assign_52Assigngenerator/G_MODEL/C/r4/1/gammasave/RestoreV2:52*
validate_shape(*1
_class'
%#loc:@generator/G_MODEL/C/r4/1/gamma*
use_locking(*
_output_shapes	
:*
T0
Ë
save/Assign_53Assigngenerator/G_MODEL/C/r4/2/betasave/RestoreV2:53*
validate_shape(*
T0*0
_class&
$"loc:@generator/G_MODEL/C/r4/2/beta*
use_locking(*
_output_shapes	
:
Í
save/Assign_54Assigngenerator/G_MODEL/C/r4/2/gammasave/RestoreV2:54*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*1
_class'
%#loc:@generator/G_MODEL/C/r4/2/gamma
ä
save/Assign_55Assign#generator/G_MODEL/C/r4/Conv/weightssave/RestoreV2:55*
validate_shape(*(
_output_shapes
:*
T0*6
_class,
*(loc:@generator/G_MODEL/C/r4/Conv/weights*
use_locking(
č
save/Assign_56Assign%generator/G_MODEL/C/r4/Conv_1/weightssave/RestoreV2:56*
validate_shape(*(
_output_shapes
:*8
_class.
,*loc:@generator/G_MODEL/C/r4/Conv_1/weights*
T0*
use_locking(
Û
save/Assign_57Assign%generator/G_MODEL/C/r4/LayerNorm/betasave/RestoreV2:57*
use_locking(*
_output_shapes	
:*
validate_shape(*8
_class.
,*loc:@generator/G_MODEL/C/r4/LayerNorm/beta*
T0
Ý
save/Assign_58Assign&generator/G_MODEL/C/r4/LayerNorm/gammasave/RestoreV2:58*
use_locking(*
validate_shape(*
T0*9
_class/
-+loc:@generator/G_MODEL/C/r4/LayerNorm/gamma*
_output_shapes	
:
Í
save/Assign_59Assigngenerator/G_MODEL/C/r4/r4/biassave/RestoreV2:59*
_output_shapes	
:*
use_locking(*1
_class'
%#loc:@generator/G_MODEL/C/r4/r4/bias*
validate_shape(*
T0
Ó
save/Assign_60Assigngenerator/G_MODEL/C/r4/r4/wsave/RestoreV2:60*'
_output_shapes
:*
T0*
use_locking(*.
_class$
" loc:@generator/G_MODEL/C/r4/r4/w*
validate_shape(
Þ
save/Assign_61Assign generator/G_MODEL/D/Conv/weightssave/RestoreV2:61*(
_output_shapes
:*
T0*
use_locking(*3
_class)
'%loc:@generator/G_MODEL/D/Conv/weights*
validate_shape(
â
save/Assign_62Assign"generator/G_MODEL/D/Conv_1/weightssave/RestoreV2:62*
validate_shape(*(
_output_shapes
:*5
_class+
)'loc:@generator/G_MODEL/D/Conv_1/weights*
use_locking(*
T0
Õ
save/Assign_63Assign"generator/G_MODEL/D/LayerNorm/betasave/RestoreV2:63*
T0*
_output_shapes	
:*
validate_shape(*5
_class+
)'loc:@generator/G_MODEL/D/LayerNorm/beta*
use_locking(
Ũ
save/Assign_64Assign#generator/G_MODEL/D/LayerNorm/gammasave/RestoreV2:64*
_output_shapes	
:*
use_locking(*
T0*6
_class,
*(loc:@generator/G_MODEL/D/LayerNorm/gamma*
validate_shape(
Ų
save/Assign_65Assign$generator/G_MODEL/D/LayerNorm_1/betasave/RestoreV2:65*
T0*7
_class-
+)loc:@generator/G_MODEL/D/LayerNorm_1/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
Û
save/Assign_66Assign%generator/G_MODEL/D/LayerNorm_1/gammasave/RestoreV2:66*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0*8
_class.
,*loc:@generator/G_MODEL/D/LayerNorm_1/gamma
Ý
save/Assign_67Assign generator/G_MODEL/E/Conv/weightssave/RestoreV2:67*
validate_shape(*3
_class)
'%loc:@generator/G_MODEL/E/Conv/weights*
use_locking(*
T0*'
_output_shapes
:@
ā
save/Assign_68Assign"generator/G_MODEL/E/Conv_1/weightssave/RestoreV2:68*
use_locking(*
T0*&
_output_shapes
:@@*
validate_shape(*5
_class+
)'loc:@generator/G_MODEL/E/Conv_1/weights
ā
save/Assign_69Assign"generator/G_MODEL/E/Conv_2/weightssave/RestoreV2:69*
T0*
validate_shape(*&
_output_shapes
:@ *5
_class+
)'loc:@generator/G_MODEL/E/Conv_2/weights*
use_locking(
Ô
save/Assign_70Assign"generator/G_MODEL/E/LayerNorm/betasave/RestoreV2:70*
_output_shapes
:@*5
_class+
)'loc:@generator/G_MODEL/E/LayerNorm/beta*
validate_shape(*
use_locking(*
T0
Ö
save/Assign_71Assign#generator/G_MODEL/E/LayerNorm/gammasave/RestoreV2:71*
T0*
_output_shapes
:@*
validate_shape(*6
_class,
*(loc:@generator/G_MODEL/E/LayerNorm/gamma*
use_locking(
Ø
save/Assign_72Assign$generator/G_MODEL/E/LayerNorm_1/betasave/RestoreV2:72*
validate_shape(*
_output_shapes
:@*
use_locking(*7
_class-
+)loc:@generator/G_MODEL/E/LayerNorm_1/beta*
T0
Ú
save/Assign_73Assign%generator/G_MODEL/E/LayerNorm_1/gammasave/RestoreV2:73*
_output_shapes
:@*8
_class.
,*loc:@generator/G_MODEL/E/LayerNorm_1/gamma*
T0*
use_locking(*
validate_shape(
Ø
save/Assign_74Assign$generator/G_MODEL/E/LayerNorm_2/betasave/RestoreV2:74*7
_class-
+)loc:@generator/G_MODEL/E/LayerNorm_2/beta*
_output_shapes
: *
use_locking(*
T0*
validate_shape(
Ú
save/Assign_75Assign%generator/G_MODEL/E/LayerNorm_2/gammasave/RestoreV2:75*
T0*
validate_shape(*8
_class.
,*loc:@generator/G_MODEL/E/LayerNorm_2/gamma*
use_locking(*
_output_shapes
: 
ė
save/Assign_76Assign(generator/G_MODEL/out_layer/Conv/weightssave/RestoreV2:76*
T0*
use_locking(*;
_class1
/-loc:@generator/G_MODEL/out_layer/Conv/weights*&
_output_shapes
: *
validate_shape(
Đ

save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_8^save/Assign_9
Ī
initNoOp(^generator/G_MODEL/A/Conv/weights/Assign*^generator/G_MODEL/A/Conv_1/weights/Assign*^generator/G_MODEL/A/Conv_2/weights/Assign*^generator/G_MODEL/A/LayerNorm/beta/Assign+^generator/G_MODEL/A/LayerNorm/gamma/Assign,^generator/G_MODEL/A/LayerNorm_1/beta/Assign-^generator/G_MODEL/A/LayerNorm_1/gamma/Assign,^generator/G_MODEL/A/LayerNorm_2/beta/Assign-^generator/G_MODEL/A/LayerNorm_2/gamma/Assign(^generator/G_MODEL/B/Conv/weights/Assign*^generator/G_MODEL/B/Conv_1/weights/Assign*^generator/G_MODEL/B/LayerNorm/beta/Assign+^generator/G_MODEL/B/LayerNorm/gamma/Assign,^generator/G_MODEL/B/LayerNorm_1/beta/Assign-^generator/G_MODEL/B/LayerNorm_1/gamma/Assign(^generator/G_MODEL/C/Conv/weights/Assign*^generator/G_MODEL/C/Conv_1/weights/Assign*^generator/G_MODEL/C/LayerNorm/beta/Assign+^generator/G_MODEL/C/LayerNorm/gamma/Assign,^generator/G_MODEL/C/LayerNorm_1/beta/Assign-^generator/G_MODEL/C/LayerNorm_1/gamma/Assign%^generator/G_MODEL/C/r1/1/beta/Assign&^generator/G_MODEL/C/r1/1/gamma/Assign%^generator/G_MODEL/C/r1/2/beta/Assign&^generator/G_MODEL/C/r1/2/gamma/Assign+^generator/G_MODEL/C/r1/Conv/weights/Assign-^generator/G_MODEL/C/r1/Conv_1/weights/Assign-^generator/G_MODEL/C/r1/LayerNorm/beta/Assign.^generator/G_MODEL/C/r1/LayerNorm/gamma/Assign&^generator/G_MODEL/C/r1/r1/bias/Assign#^generator/G_MODEL/C/r1/r1/w/Assign%^generator/G_MODEL/C/r2/1/beta/Assign&^generator/G_MODEL/C/r2/1/gamma/Assign%^generator/G_MODEL/C/r2/2/beta/Assign&^generator/G_MODEL/C/r2/2/gamma/Assign+^generator/G_MODEL/C/r2/Conv/weights/Assign-^generator/G_MODEL/C/r2/Conv_1/weights/Assign-^generator/G_MODEL/C/r2/LayerNorm/beta/Assign.^generator/G_MODEL/C/r2/LayerNorm/gamma/Assign&^generator/G_MODEL/C/r2/r2/bias/Assign#^generator/G_MODEL/C/r2/r2/w/Assign%^generator/G_MODEL/C/r3/1/beta/Assign&^generator/G_MODEL/C/r3/1/gamma/Assign%^generator/G_MODEL/C/r3/2/beta/Assign&^generator/G_MODEL/C/r3/2/gamma/Assign+^generator/G_MODEL/C/r3/Conv/weights/Assign-^generator/G_MODEL/C/r3/Conv_1/weights/Assign-^generator/G_MODEL/C/r3/LayerNorm/beta/Assign.^generator/G_MODEL/C/r3/LayerNorm/gamma/Assign&^generator/G_MODEL/C/r3/r3/bias/Assign#^generator/G_MODEL/C/r3/r3/w/Assign%^generator/G_MODEL/C/r4/1/beta/Assign&^generator/G_MODEL/C/r4/1/gamma/Assign%^generator/G_MODEL/C/r4/2/beta/Assign&^generator/G_MODEL/C/r4/2/gamma/Assign+^generator/G_MODEL/C/r4/Conv/weights/Assign-^generator/G_MODEL/C/r4/Conv_1/weights/Assign-^generator/G_MODEL/C/r4/LayerNorm/beta/Assign.^generator/G_MODEL/C/r4/LayerNorm/gamma/Assign&^generator/G_MODEL/C/r4/r4/bias/Assign#^generator/G_MODEL/C/r4/r4/w/Assign(^generator/G_MODEL/D/Conv/weights/Assign*^generator/G_MODEL/D/Conv_1/weights/Assign*^generator/G_MODEL/D/LayerNorm/beta/Assign+^generator/G_MODEL/D/LayerNorm/gamma/Assign,^generator/G_MODEL/D/LayerNorm_1/beta/Assign-^generator/G_MODEL/D/LayerNorm_1/gamma/Assign(^generator/G_MODEL/E/Conv/weights/Assign*^generator/G_MODEL/E/Conv_1/weights/Assign*^generator/G_MODEL/E/Conv_2/weights/Assign*^generator/G_MODEL/E/LayerNorm/beta/Assign+^generator/G_MODEL/E/LayerNorm/gamma/Assign,^generator/G_MODEL/E/LayerNorm_1/beta/Assign-^generator/G_MODEL/E/LayerNorm_1/gamma/Assign,^generator/G_MODEL/E/LayerNorm_2/beta/Assign-^generator/G_MODEL/E/LayerNorm_2/gamma/Assign0^generator/G_MODEL/out_layer/Conv/weights/Assign
[
save/filename_1/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save/filename_1PlaceholderWithDefaultsave/filename_1/input*
_output_shapes
: *
shape: *
dtype0
i
save/Const_1PlaceholderWithDefaultsave/filename_1*
dtype0*
_output_shapes
: *
shape: 

save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_c90b4e1bb6f644f3a9010925ac7efe64/part*
_output_shapes
: 
w
save/StringJoin
StringJoinsave/Const_1save/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
Ģ
save/SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:M*Å
valueŧBļMB generator/G_MODEL/A/Conv/weightsB"generator/G_MODEL/A/Conv_1/weightsB"generator/G_MODEL/A/Conv_2/weightsB"generator/G_MODEL/A/LayerNorm/betaB#generator/G_MODEL/A/LayerNorm/gammaB$generator/G_MODEL/A/LayerNorm_1/betaB%generator/G_MODEL/A/LayerNorm_1/gammaB$generator/G_MODEL/A/LayerNorm_2/betaB%generator/G_MODEL/A/LayerNorm_2/gammaB generator/G_MODEL/B/Conv/weightsB"generator/G_MODEL/B/Conv_1/weightsB"generator/G_MODEL/B/LayerNorm/betaB#generator/G_MODEL/B/LayerNorm/gammaB$generator/G_MODEL/B/LayerNorm_1/betaB%generator/G_MODEL/B/LayerNorm_1/gammaB generator/G_MODEL/C/Conv/weightsB"generator/G_MODEL/C/Conv_1/weightsB"generator/G_MODEL/C/LayerNorm/betaB#generator/G_MODEL/C/LayerNorm/gammaB$generator/G_MODEL/C/LayerNorm_1/betaB%generator/G_MODEL/C/LayerNorm_1/gammaBgenerator/G_MODEL/C/r1/1/betaBgenerator/G_MODEL/C/r1/1/gammaBgenerator/G_MODEL/C/r1/2/betaBgenerator/G_MODEL/C/r1/2/gammaB#generator/G_MODEL/C/r1/Conv/weightsB%generator/G_MODEL/C/r1/Conv_1/weightsB%generator/G_MODEL/C/r1/LayerNorm/betaB&generator/G_MODEL/C/r1/LayerNorm/gammaBgenerator/G_MODEL/C/r1/r1/biasBgenerator/G_MODEL/C/r1/r1/wBgenerator/G_MODEL/C/r2/1/betaBgenerator/G_MODEL/C/r2/1/gammaBgenerator/G_MODEL/C/r2/2/betaBgenerator/G_MODEL/C/r2/2/gammaB#generator/G_MODEL/C/r2/Conv/weightsB%generator/G_MODEL/C/r2/Conv_1/weightsB%generator/G_MODEL/C/r2/LayerNorm/betaB&generator/G_MODEL/C/r2/LayerNorm/gammaBgenerator/G_MODEL/C/r2/r2/biasBgenerator/G_MODEL/C/r2/r2/wBgenerator/G_MODEL/C/r3/1/betaBgenerator/G_MODEL/C/r3/1/gammaBgenerator/G_MODEL/C/r3/2/betaBgenerator/G_MODEL/C/r3/2/gammaB#generator/G_MODEL/C/r3/Conv/weightsB%generator/G_MODEL/C/r3/Conv_1/weightsB%generator/G_MODEL/C/r3/LayerNorm/betaB&generator/G_MODEL/C/r3/LayerNorm/gammaBgenerator/G_MODEL/C/r3/r3/biasBgenerator/G_MODEL/C/r3/r3/wBgenerator/G_MODEL/C/r4/1/betaBgenerator/G_MODEL/C/r4/1/gammaBgenerator/G_MODEL/C/r4/2/betaBgenerator/G_MODEL/C/r4/2/gammaB#generator/G_MODEL/C/r4/Conv/weightsB%generator/G_MODEL/C/r4/Conv_1/weightsB%generator/G_MODEL/C/r4/LayerNorm/betaB&generator/G_MODEL/C/r4/LayerNorm/gammaBgenerator/G_MODEL/C/r4/r4/biasBgenerator/G_MODEL/C/r4/r4/wB generator/G_MODEL/D/Conv/weightsB"generator/G_MODEL/D/Conv_1/weightsB"generator/G_MODEL/D/LayerNorm/betaB#generator/G_MODEL/D/LayerNorm/gammaB$generator/G_MODEL/D/LayerNorm_1/betaB%generator/G_MODEL/D/LayerNorm_1/gammaB generator/G_MODEL/E/Conv/weightsB"generator/G_MODEL/E/Conv_1/weightsB"generator/G_MODEL/E/Conv_2/weightsB"generator/G_MODEL/E/LayerNorm/betaB#generator/G_MODEL/E/LayerNorm/gammaB$generator/G_MODEL/E/LayerNorm_1/betaB%generator/G_MODEL/E/LayerNorm_1/gammaB$generator/G_MODEL/E/LayerNorm_2/betaB%generator/G_MODEL/E/LayerNorm_2/gammaB(generator/G_MODEL/out_layer/Conv/weights

save/SaveV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*Ŋ
valueĨBĒMB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:M

save/SaveV2_1SaveV2save/ShardedFilenamesave/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slices generator/G_MODEL/A/Conv/weights"generator/G_MODEL/A/Conv_1/weights"generator/G_MODEL/A/Conv_2/weights"generator/G_MODEL/A/LayerNorm/beta#generator/G_MODEL/A/LayerNorm/gamma$generator/G_MODEL/A/LayerNorm_1/beta%generator/G_MODEL/A/LayerNorm_1/gamma$generator/G_MODEL/A/LayerNorm_2/beta%generator/G_MODEL/A/LayerNorm_2/gamma generator/G_MODEL/B/Conv/weights"generator/G_MODEL/B/Conv_1/weights"generator/G_MODEL/B/LayerNorm/beta#generator/G_MODEL/B/LayerNorm/gamma$generator/G_MODEL/B/LayerNorm_1/beta%generator/G_MODEL/B/LayerNorm_1/gamma generator/G_MODEL/C/Conv/weights"generator/G_MODEL/C/Conv_1/weights"generator/G_MODEL/C/LayerNorm/beta#generator/G_MODEL/C/LayerNorm/gamma$generator/G_MODEL/C/LayerNorm_1/beta%generator/G_MODEL/C/LayerNorm_1/gammagenerator/G_MODEL/C/r1/1/betagenerator/G_MODEL/C/r1/1/gammagenerator/G_MODEL/C/r1/2/betagenerator/G_MODEL/C/r1/2/gamma#generator/G_MODEL/C/r1/Conv/weights%generator/G_MODEL/C/r1/Conv_1/weights%generator/G_MODEL/C/r1/LayerNorm/beta&generator/G_MODEL/C/r1/LayerNorm/gammagenerator/G_MODEL/C/r1/r1/biasgenerator/G_MODEL/C/r1/r1/wgenerator/G_MODEL/C/r2/1/betagenerator/G_MODEL/C/r2/1/gammagenerator/G_MODEL/C/r2/2/betagenerator/G_MODEL/C/r2/2/gamma#generator/G_MODEL/C/r2/Conv/weights%generator/G_MODEL/C/r2/Conv_1/weights%generator/G_MODEL/C/r2/LayerNorm/beta&generator/G_MODEL/C/r2/LayerNorm/gammagenerator/G_MODEL/C/r2/r2/biasgenerator/G_MODEL/C/r2/r2/wgenerator/G_MODEL/C/r3/1/betagenerator/G_MODEL/C/r3/1/gammagenerator/G_MODEL/C/r3/2/betagenerator/G_MODEL/C/r3/2/gamma#generator/G_MODEL/C/r3/Conv/weights%generator/G_MODEL/C/r3/Conv_1/weights%generator/G_MODEL/C/r3/LayerNorm/beta&generator/G_MODEL/C/r3/LayerNorm/gammagenerator/G_MODEL/C/r3/r3/biasgenerator/G_MODEL/C/r3/r3/wgenerator/G_MODEL/C/r4/1/betagenerator/G_MODEL/C/r4/1/gammagenerator/G_MODEL/C/r4/2/betagenerator/G_MODEL/C/r4/2/gamma#generator/G_MODEL/C/r4/Conv/weights%generator/G_MODEL/C/r4/Conv_1/weights%generator/G_MODEL/C/r4/LayerNorm/beta&generator/G_MODEL/C/r4/LayerNorm/gammagenerator/G_MODEL/C/r4/r4/biasgenerator/G_MODEL/C/r4/r4/w generator/G_MODEL/D/Conv/weights"generator/G_MODEL/D/Conv_1/weights"generator/G_MODEL/D/LayerNorm/beta#generator/G_MODEL/D/LayerNorm/gamma$generator/G_MODEL/D/LayerNorm_1/beta%generator/G_MODEL/D/LayerNorm_1/gamma generator/G_MODEL/E/Conv/weights"generator/G_MODEL/E/Conv_1/weights"generator/G_MODEL/E/Conv_2/weights"generator/G_MODEL/E/LayerNorm/beta#generator/G_MODEL/E/LayerNorm/gamma$generator/G_MODEL/E/LayerNorm_1/beta%generator/G_MODEL/E/LayerNorm_1/gamma$generator/G_MODEL/E/LayerNorm_2/beta%generator/G_MODEL/E/LayerNorm_2/gamma(generator/G_MODEL/out_layer/Conv/weights"/device:CPU:0*[
dtypesQ
O2M
Ī
save/control_dependency_1Identitysave/ShardedFilename^save/SaveV2_1"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Ū
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency_1"/device:CPU:0*
N*

axis *
_output_shapes
:*
T0

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixessave/Const_1"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentitysave/Const_1^save/MergeV2Checkpoints^save/control_dependency_1"/device:CPU:0*
T0*
_output_shapes
: 
Ķ
save/RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:M*Å
valueŧBļMB generator/G_MODEL/A/Conv/weightsB"generator/G_MODEL/A/Conv_1/weightsB"generator/G_MODEL/A/Conv_2/weightsB"generator/G_MODEL/A/LayerNorm/betaB#generator/G_MODEL/A/LayerNorm/gammaB$generator/G_MODEL/A/LayerNorm_1/betaB%generator/G_MODEL/A/LayerNorm_1/gammaB$generator/G_MODEL/A/LayerNorm_2/betaB%generator/G_MODEL/A/LayerNorm_2/gammaB generator/G_MODEL/B/Conv/weightsB"generator/G_MODEL/B/Conv_1/weightsB"generator/G_MODEL/B/LayerNorm/betaB#generator/G_MODEL/B/LayerNorm/gammaB$generator/G_MODEL/B/LayerNorm_1/betaB%generator/G_MODEL/B/LayerNorm_1/gammaB generator/G_MODEL/C/Conv/weightsB"generator/G_MODEL/C/Conv_1/weightsB"generator/G_MODEL/C/LayerNorm/betaB#generator/G_MODEL/C/LayerNorm/gammaB$generator/G_MODEL/C/LayerNorm_1/betaB%generator/G_MODEL/C/LayerNorm_1/gammaBgenerator/G_MODEL/C/r1/1/betaBgenerator/G_MODEL/C/r1/1/gammaBgenerator/G_MODEL/C/r1/2/betaBgenerator/G_MODEL/C/r1/2/gammaB#generator/G_MODEL/C/r1/Conv/weightsB%generator/G_MODEL/C/r1/Conv_1/weightsB%generator/G_MODEL/C/r1/LayerNorm/betaB&generator/G_MODEL/C/r1/LayerNorm/gammaBgenerator/G_MODEL/C/r1/r1/biasBgenerator/G_MODEL/C/r1/r1/wBgenerator/G_MODEL/C/r2/1/betaBgenerator/G_MODEL/C/r2/1/gammaBgenerator/G_MODEL/C/r2/2/betaBgenerator/G_MODEL/C/r2/2/gammaB#generator/G_MODEL/C/r2/Conv/weightsB%generator/G_MODEL/C/r2/Conv_1/weightsB%generator/G_MODEL/C/r2/LayerNorm/betaB&generator/G_MODEL/C/r2/LayerNorm/gammaBgenerator/G_MODEL/C/r2/r2/biasBgenerator/G_MODEL/C/r2/r2/wBgenerator/G_MODEL/C/r3/1/betaBgenerator/G_MODEL/C/r3/1/gammaBgenerator/G_MODEL/C/r3/2/betaBgenerator/G_MODEL/C/r3/2/gammaB#generator/G_MODEL/C/r3/Conv/weightsB%generator/G_MODEL/C/r3/Conv_1/weightsB%generator/G_MODEL/C/r3/LayerNorm/betaB&generator/G_MODEL/C/r3/LayerNorm/gammaBgenerator/G_MODEL/C/r3/r3/biasBgenerator/G_MODEL/C/r3/r3/wBgenerator/G_MODEL/C/r4/1/betaBgenerator/G_MODEL/C/r4/1/gammaBgenerator/G_MODEL/C/r4/2/betaBgenerator/G_MODEL/C/r4/2/gammaB#generator/G_MODEL/C/r4/Conv/weightsB%generator/G_MODEL/C/r4/Conv_1/weightsB%generator/G_MODEL/C/r4/LayerNorm/betaB&generator/G_MODEL/C/r4/LayerNorm/gammaBgenerator/G_MODEL/C/r4/r4/biasBgenerator/G_MODEL/C/r4/r4/wB generator/G_MODEL/D/Conv/weightsB"generator/G_MODEL/D/Conv_1/weightsB"generator/G_MODEL/D/LayerNorm/betaB#generator/G_MODEL/D/LayerNorm/gammaB$generator/G_MODEL/D/LayerNorm_1/betaB%generator/G_MODEL/D/LayerNorm_1/gammaB generator/G_MODEL/E/Conv/weightsB"generator/G_MODEL/E/Conv_1/weightsB"generator/G_MODEL/E/Conv_2/weightsB"generator/G_MODEL/E/LayerNorm/betaB#generator/G_MODEL/E/LayerNorm/gammaB$generator/G_MODEL/E/LayerNorm_1/betaB%generator/G_MODEL/E/LayerNorm_1/gammaB$generator/G_MODEL/E/LayerNorm_2/betaB%generator/G_MODEL/E/LayerNorm_2/gammaB(generator/G_MODEL/out_layer/Conv/weights*
dtype0

!save/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*Ŋ
valueĨBĒMB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:M*
dtype0
Ķ
save/RestoreV2_1	RestoreV2save/Const_1save/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices"/device:CPU:0*[
dtypesQ
O2M*Ę
_output_shapes·
ī:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Û
save/Assign_77Assign generator/G_MODEL/A/Conv/weightssave/RestoreV2_1*&
_output_shapes
: *
validate_shape(*
T0*3
_class)
'%loc:@generator/G_MODEL/A/Conv/weights*
use_locking(
á
save/Assign_78Assign"generator/G_MODEL/A/Conv_1/weightssave/RestoreV2_1:1*
T0*
validate_shape(*
use_locking(*&
_output_shapes
: @*5
_class+
)'loc:@generator/G_MODEL/A/Conv_1/weights
á
save/Assign_79Assign"generator/G_MODEL/A/Conv_2/weightssave/RestoreV2_1:2*
validate_shape(*
use_locking(*
T0*&
_output_shapes
:@@*5
_class+
)'loc:@generator/G_MODEL/A/Conv_2/weights
Õ
save/Assign_80Assign"generator/G_MODEL/A/LayerNorm/betasave/RestoreV2_1:3*
T0*
use_locking(*
validate_shape(*
_output_shapes
: *5
_class+
)'loc:@generator/G_MODEL/A/LayerNorm/beta
Ũ
save/Assign_81Assign#generator/G_MODEL/A/LayerNorm/gammasave/RestoreV2_1:4*
_output_shapes
: *6
_class,
*(loc:@generator/G_MODEL/A/LayerNorm/gamma*
validate_shape(*
T0*
use_locking(
Ų
save/Assign_82Assign$generator/G_MODEL/A/LayerNorm_1/betasave/RestoreV2_1:5*
validate_shape(*7
_class-
+)loc:@generator/G_MODEL/A/LayerNorm_1/beta*
use_locking(*
_output_shapes
:@*
T0
Û
save/Assign_83Assign%generator/G_MODEL/A/LayerNorm_1/gammasave/RestoreV2_1:6*
validate_shape(*
T0*8
_class.
,*loc:@generator/G_MODEL/A/LayerNorm_1/gamma*
use_locking(*
_output_shapes
:@
Ų
save/Assign_84Assign$generator/G_MODEL/A/LayerNorm_2/betasave/RestoreV2_1:7*7
_class-
+)loc:@generator/G_MODEL/A/LayerNorm_2/beta*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(
Û
save/Assign_85Assign%generator/G_MODEL/A/LayerNorm_2/gammasave/RestoreV2_1:8*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@*8
_class.
,*loc:@generator/G_MODEL/A/LayerNorm_2/gamma
Þ
save/Assign_86Assign generator/G_MODEL/B/Conv/weightssave/RestoreV2_1:9*
T0*
use_locking(*
validate_shape(*'
_output_shapes
:@*3
_class)
'%loc:@generator/G_MODEL/B/Conv/weights
ä
save/Assign_87Assign"generator/G_MODEL/B/Conv_1/weightssave/RestoreV2_1:10*(
_output_shapes
:*
T0*5
_class+
)'loc:@generator/G_MODEL/B/Conv_1/weights*
validate_shape(*
use_locking(
Ũ
save/Assign_88Assign"generator/G_MODEL/B/LayerNorm/betasave/RestoreV2_1:11*
use_locking(*5
_class+
)'loc:@generator/G_MODEL/B/LayerNorm/beta*
T0*
_output_shapes	
:*
validate_shape(
Ų
save/Assign_89Assign#generator/G_MODEL/B/LayerNorm/gammasave/RestoreV2_1:12*
validate_shape(*6
_class,
*(loc:@generator/G_MODEL/B/LayerNorm/gamma*
T0*
use_locking(*
_output_shapes	
:
Û
save/Assign_90Assign$generator/G_MODEL/B/LayerNorm_1/betasave/RestoreV2_1:13*
use_locking(*7
_class-
+)loc:@generator/G_MODEL/B/LayerNorm_1/beta*
_output_shapes	
:*
validate_shape(*
T0
Ý
save/Assign_91Assign%generator/G_MODEL/B/LayerNorm_1/gammasave/RestoreV2_1:14*
validate_shape(*
use_locking(*8
_class.
,*loc:@generator/G_MODEL/B/LayerNorm_1/gamma*
T0*
_output_shapes	
:
ā
save/Assign_92Assign generator/G_MODEL/C/Conv/weightssave/RestoreV2_1:15*3
_class)
'%loc:@generator/G_MODEL/C/Conv/weights*(
_output_shapes
:*
T0*
validate_shape(*
use_locking(
ä
save/Assign_93Assign"generator/G_MODEL/C/Conv_1/weightssave/RestoreV2_1:16*5
_class+
)'loc:@generator/G_MODEL/C/Conv_1/weights*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Ũ
save/Assign_94Assign"generator/G_MODEL/C/LayerNorm/betasave/RestoreV2_1:17*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(*5
_class+
)'loc:@generator/G_MODEL/C/LayerNorm/beta
Ų
save/Assign_95Assign#generator/G_MODEL/C/LayerNorm/gammasave/RestoreV2_1:18*
validate_shape(*
use_locking(*
_output_shapes	
:*6
_class,
*(loc:@generator/G_MODEL/C/LayerNorm/gamma*
T0
Û
save/Assign_96Assign$generator/G_MODEL/C/LayerNorm_1/betasave/RestoreV2_1:19*
_output_shapes	
:*
use_locking(*7
_class-
+)loc:@generator/G_MODEL/C/LayerNorm_1/beta*
T0*
validate_shape(
Ý
save/Assign_97Assign%generator/G_MODEL/C/LayerNorm_1/gammasave/RestoreV2_1:20*
validate_shape(*
use_locking(*8
_class.
,*loc:@generator/G_MODEL/C/LayerNorm_1/gamma*
T0*
_output_shapes	
:
Í
save/Assign_98Assigngenerator/G_MODEL/C/r1/1/betasave/RestoreV2_1:21*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(*0
_class&
$"loc:@generator/G_MODEL/C/r1/1/beta
Ï
save/Assign_99Assigngenerator/G_MODEL/C/r1/1/gammasave/RestoreV2_1:22*1
_class'
%#loc:@generator/G_MODEL/C/r1/1/gamma*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
Î
save/Assign_100Assigngenerator/G_MODEL/C/r1/2/betasave/RestoreV2_1:23*0
_class&
$"loc:@generator/G_MODEL/C/r1/2/beta*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:
Ð
save/Assign_101Assigngenerator/G_MODEL/C/r1/2/gammasave/RestoreV2_1:24*
_output_shapes	
:*1
_class'
%#loc:@generator/G_MODEL/C/r1/2/gamma*
use_locking(*
T0*
validate_shape(
į
save/Assign_102Assign#generator/G_MODEL/C/r1/Conv/weightssave/RestoreV2_1:25*
use_locking(*
T0*
validate_shape(*(
_output_shapes
:*6
_class,
*(loc:@generator/G_MODEL/C/r1/Conv/weights
ë
save/Assign_103Assign%generator/G_MODEL/C/r1/Conv_1/weightssave/RestoreV2_1:26*8
_class.
,*loc:@generator/G_MODEL/C/r1/Conv_1/weights*
use_locking(*
T0*
validate_shape(*(
_output_shapes
:
Þ
save/Assign_104Assign%generator/G_MODEL/C/r1/LayerNorm/betasave/RestoreV2_1:27*
validate_shape(*
T0*8
_class.
,*loc:@generator/G_MODEL/C/r1/LayerNorm/beta*
_output_shapes	
:*
use_locking(
ā
save/Assign_105Assign&generator/G_MODEL/C/r1/LayerNorm/gammasave/RestoreV2_1:28*
_output_shapes	
:*
T0*
use_locking(*9
_class/
-+loc:@generator/G_MODEL/C/r1/LayerNorm/gamma*
validate_shape(
Ð
save/Assign_106Assigngenerator/G_MODEL/C/r1/r1/biassave/RestoreV2_1:29*
T0*
_output_shapes	
:*1
_class'
%#loc:@generator/G_MODEL/C/r1/r1/bias*
validate_shape(*
use_locking(
Ö
save/Assign_107Assigngenerator/G_MODEL/C/r1/r1/wsave/RestoreV2_1:30*
use_locking(*
T0*'
_output_shapes
:*
validate_shape(*.
_class$
" loc:@generator/G_MODEL/C/r1/r1/w
Î
save/Assign_108Assigngenerator/G_MODEL/C/r2/1/betasave/RestoreV2_1:31*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0*0
_class&
$"loc:@generator/G_MODEL/C/r2/1/beta
Ð
save/Assign_109Assigngenerator/G_MODEL/C/r2/1/gammasave/RestoreV2_1:32*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(*1
_class'
%#loc:@generator/G_MODEL/C/r2/1/gamma
Î
save/Assign_110Assigngenerator/G_MODEL/C/r2/2/betasave/RestoreV2_1:33*
T0*
use_locking(*
_output_shapes	
:*0
_class&
$"loc:@generator/G_MODEL/C/r2/2/beta*
validate_shape(
Ð
save/Assign_111Assigngenerator/G_MODEL/C/r2/2/gammasave/RestoreV2_1:34*
T0*1
_class'
%#loc:@generator/G_MODEL/C/r2/2/gamma*
use_locking(*
_output_shapes	
:*
validate_shape(
į
save/Assign_112Assign#generator/G_MODEL/C/r2/Conv/weightssave/RestoreV2_1:35*6
_class,
*(loc:@generator/G_MODEL/C/r2/Conv/weights*
use_locking(*
validate_shape(*(
_output_shapes
:*
T0
ë
save/Assign_113Assign%generator/G_MODEL/C/r2/Conv_1/weightssave/RestoreV2_1:36*
validate_shape(*8
_class.
,*loc:@generator/G_MODEL/C/r2/Conv_1/weights*(
_output_shapes
:*
T0*
use_locking(
Þ
save/Assign_114Assign%generator/G_MODEL/C/r2/LayerNorm/betasave/RestoreV2_1:37*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@generator/G_MODEL/C/r2/LayerNorm/beta
ā
save/Assign_115Assign&generator/G_MODEL/C/r2/LayerNorm/gammasave/RestoreV2_1:38*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0*9
_class/
-+loc:@generator/G_MODEL/C/r2/LayerNorm/gamma
Ð
save/Assign_116Assigngenerator/G_MODEL/C/r2/r2/biassave/RestoreV2_1:39*
T0*1
_class'
%#loc:@generator/G_MODEL/C/r2/r2/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
Ö
save/Assign_117Assigngenerator/G_MODEL/C/r2/r2/wsave/RestoreV2_1:40*'
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@generator/G_MODEL/C/r2/r2/w*
validate_shape(
Î
save/Assign_118Assigngenerator/G_MODEL/C/r3/1/betasave/RestoreV2_1:41*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0*0
_class&
$"loc:@generator/G_MODEL/C/r3/1/beta
Ð
save/Assign_119Assigngenerator/G_MODEL/C/r3/1/gammasave/RestoreV2_1:42*1
_class'
%#loc:@generator/G_MODEL/C/r3/1/gamma*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(
Î
save/Assign_120Assigngenerator/G_MODEL/C/r3/2/betasave/RestoreV2_1:43*0
_class&
$"loc:@generator/G_MODEL/C/r3/2/beta*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
Ð
save/Assign_121Assigngenerator/G_MODEL/C/r3/2/gammasave/RestoreV2_1:44*
T0*1
_class'
%#loc:@generator/G_MODEL/C/r3/2/gamma*
_output_shapes	
:*
use_locking(*
validate_shape(
į
save/Assign_122Assign#generator/G_MODEL/C/r3/Conv/weightssave/RestoreV2_1:45*(
_output_shapes
:*
use_locking(*6
_class,
*(loc:@generator/G_MODEL/C/r3/Conv/weights*
validate_shape(*
T0
ë
save/Assign_123Assign%generator/G_MODEL/C/r3/Conv_1/weightssave/RestoreV2_1:46*
T0*(
_output_shapes
:*8
_class.
,*loc:@generator/G_MODEL/C/r3/Conv_1/weights*
use_locking(*
validate_shape(
Þ
save/Assign_124Assign%generator/G_MODEL/C/r3/LayerNorm/betasave/RestoreV2_1:47*
use_locking(*
validate_shape(*8
_class.
,*loc:@generator/G_MODEL/C/r3/LayerNorm/beta*
_output_shapes	
:*
T0
ā
save/Assign_125Assign&generator/G_MODEL/C/r3/LayerNorm/gammasave/RestoreV2_1:48*
validate_shape(*
T0*9
_class/
-+loc:@generator/G_MODEL/C/r3/LayerNorm/gamma*
_output_shapes	
:*
use_locking(
Ð
save/Assign_126Assigngenerator/G_MODEL/C/r3/r3/biassave/RestoreV2_1:49*
validate_shape(*
use_locking(*
T0*1
_class'
%#loc:@generator/G_MODEL/C/r3/r3/bias*
_output_shapes	
:
Ö
save/Assign_127Assigngenerator/G_MODEL/C/r3/r3/wsave/RestoreV2_1:50*'
_output_shapes
:*.
_class$
" loc:@generator/G_MODEL/C/r3/r3/w*
T0*
use_locking(*
validate_shape(
Î
save/Assign_128Assigngenerator/G_MODEL/C/r4/1/betasave/RestoreV2_1:51*
_output_shapes	
:*0
_class&
$"loc:@generator/G_MODEL/C/r4/1/beta*
validate_shape(*
use_locking(*
T0
Ð
save/Assign_129Assigngenerator/G_MODEL/C/r4/1/gammasave/RestoreV2_1:52*1
_class'
%#loc:@generator/G_MODEL/C/r4/1/gamma*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
Î
save/Assign_130Assigngenerator/G_MODEL/C/r4/2/betasave/RestoreV2_1:53*
_output_shapes	
:*
T0*0
_class&
$"loc:@generator/G_MODEL/C/r4/2/beta*
use_locking(*
validate_shape(
Ð
save/Assign_131Assigngenerator/G_MODEL/C/r4/2/gammasave/RestoreV2_1:54*
use_locking(*
validate_shape(*
T0*1
_class'
%#loc:@generator/G_MODEL/C/r4/2/gamma*
_output_shapes	
:
į
save/Assign_132Assign#generator/G_MODEL/C/r4/Conv/weightssave/RestoreV2_1:55*
validate_shape(*
T0*
use_locking(*6
_class,
*(loc:@generator/G_MODEL/C/r4/Conv/weights*(
_output_shapes
:
ë
save/Assign_133Assign%generator/G_MODEL/C/r4/Conv_1/weightssave/RestoreV2_1:56*
T0*8
_class.
,*loc:@generator/G_MODEL/C/r4/Conv_1/weights*
validate_shape(*
use_locking(*(
_output_shapes
:
Þ
save/Assign_134Assign%generator/G_MODEL/C/r4/LayerNorm/betasave/RestoreV2_1:57*
_output_shapes	
:*8
_class.
,*loc:@generator/G_MODEL/C/r4/LayerNorm/beta*
T0*
validate_shape(*
use_locking(
ā
save/Assign_135Assign&generator/G_MODEL/C/r4/LayerNorm/gammasave/RestoreV2_1:58*
validate_shape(*
_output_shapes	
:*9
_class/
-+loc:@generator/G_MODEL/C/r4/LayerNorm/gamma*
T0*
use_locking(
Ð
save/Assign_136Assigngenerator/G_MODEL/C/r4/r4/biassave/RestoreV2_1:59*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*1
_class'
%#loc:@generator/G_MODEL/C/r4/r4/bias
Ö
save/Assign_137Assigngenerator/G_MODEL/C/r4/r4/wsave/RestoreV2_1:60*'
_output_shapes
:*
validate_shape(*
T0*
use_locking(*.
_class$
" loc:@generator/G_MODEL/C/r4/r4/w
á
save/Assign_138Assign generator/G_MODEL/D/Conv/weightssave/RestoreV2_1:61*(
_output_shapes
:*3
_class)
'%loc:@generator/G_MODEL/D/Conv/weights*
use_locking(*
T0*
validate_shape(
å
save/Assign_139Assign"generator/G_MODEL/D/Conv_1/weightssave/RestoreV2_1:62*(
_output_shapes
:*
T0*5
_class+
)'loc:@generator/G_MODEL/D/Conv_1/weights*
use_locking(*
validate_shape(
Ø
save/Assign_140Assign"generator/G_MODEL/D/LayerNorm/betasave/RestoreV2_1:63*
T0*
_output_shapes	
:*5
_class+
)'loc:@generator/G_MODEL/D/LayerNorm/beta*
validate_shape(*
use_locking(
Ú
save/Assign_141Assign#generator/G_MODEL/D/LayerNorm/gammasave/RestoreV2_1:64*6
_class,
*(loc:@generator/G_MODEL/D/LayerNorm/gamma*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(
Ü
save/Assign_142Assign$generator/G_MODEL/D/LayerNorm_1/betasave/RestoreV2_1:65*
validate_shape(*
T0*
_output_shapes	
:*7
_class-
+)loc:@generator/G_MODEL/D/LayerNorm_1/beta*
use_locking(
Þ
save/Assign_143Assign%generator/G_MODEL/D/LayerNorm_1/gammasave/RestoreV2_1:66*
use_locking(*8
_class.
,*loc:@generator/G_MODEL/D/LayerNorm_1/gamma*
T0*
_output_shapes	
:*
validate_shape(
ā
save/Assign_144Assign generator/G_MODEL/E/Conv/weightssave/RestoreV2_1:67*
T0*3
_class)
'%loc:@generator/G_MODEL/E/Conv/weights*'
_output_shapes
:@*
validate_shape(*
use_locking(
ã
save/Assign_145Assign"generator/G_MODEL/E/Conv_1/weightssave/RestoreV2_1:68*
T0*&
_output_shapes
:@@*
use_locking(*
validate_shape(*5
_class+
)'loc:@generator/G_MODEL/E/Conv_1/weights
ã
save/Assign_146Assign"generator/G_MODEL/E/Conv_2/weightssave/RestoreV2_1:69*5
_class+
)'loc:@generator/G_MODEL/E/Conv_2/weights*
use_locking(*
validate_shape(*&
_output_shapes
:@ *
T0
Ũ
save/Assign_147Assign"generator/G_MODEL/E/LayerNorm/betasave/RestoreV2_1:70*
validate_shape(*
T0*5
_class+
)'loc:@generator/G_MODEL/E/LayerNorm/beta*
_output_shapes
:@*
use_locking(
Ų
save/Assign_148Assign#generator/G_MODEL/E/LayerNorm/gammasave/RestoreV2_1:71*
use_locking(*
_output_shapes
:@*
T0*6
_class,
*(loc:@generator/G_MODEL/E/LayerNorm/gamma*
validate_shape(
Û
save/Assign_149Assign$generator/G_MODEL/E/LayerNorm_1/betasave/RestoreV2_1:72*7
_class-
+)loc:@generator/G_MODEL/E/LayerNorm_1/beta*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(
Ý
save/Assign_150Assign%generator/G_MODEL/E/LayerNorm_1/gammasave/RestoreV2_1:73*
_output_shapes
:@*
T0*8
_class.
,*loc:@generator/G_MODEL/E/LayerNorm_1/gamma*
use_locking(*
validate_shape(
Û
save/Assign_151Assign$generator/G_MODEL/E/LayerNorm_2/betasave/RestoreV2_1:74*
_output_shapes
: *
validate_shape(*
T0*7
_class-
+)loc:@generator/G_MODEL/E/LayerNorm_2/beta*
use_locking(
Ý
save/Assign_152Assign%generator/G_MODEL/E/LayerNorm_2/gammasave/RestoreV2_1:75*8
_class.
,*loc:@generator/G_MODEL/E/LayerNorm_2/gamma*
_output_shapes
: *
validate_shape(*
T0*
use_locking(
ï
save/Assign_153Assign(generator/G_MODEL/out_layer/Conv/weightssave/RestoreV2_1:76*&
_output_shapes
: *;
_class1
/-loc:@generator/G_MODEL/out_layer/Conv/weights*
T0*
use_locking(*
validate_shape(
í

save/restore_shardNoOp^save/Assign_100^save/Assign_101^save/Assign_102^save/Assign_103^save/Assign_104^save/Assign_105^save/Assign_106^save/Assign_107^save/Assign_108^save/Assign_109^save/Assign_110^save/Assign_111^save/Assign_112^save/Assign_113^save/Assign_114^save/Assign_115^save/Assign_116^save/Assign_117^save/Assign_118^save/Assign_119^save/Assign_120^save/Assign_121^save/Assign_122^save/Assign_123^save/Assign_124^save/Assign_125^save/Assign_126^save/Assign_127^save/Assign_128^save/Assign_129^save/Assign_130^save/Assign_131^save/Assign_132^save/Assign_133^save/Assign_134^save/Assign_135^save/Assign_136^save/Assign_137^save/Assign_138^save/Assign_139^save/Assign_140^save/Assign_141^save/Assign_142^save/Assign_143^save/Assign_144^save/Assign_145^save/Assign_146^save/Assign_147^save/Assign_148^save/Assign_149^save/Assign_150^save/Assign_151^save/Assign_152^save/Assign_153^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99
/
save/restore_all_1NoOp^save/restore_shard"@
save/Const_1:0save/Identity:0save/restore_all_1 (5 @F8"Ąp
	variablespp
đ
"generator/G_MODEL/A/Conv/weights:0'generator/G_MODEL/A/Conv/weights/Assign'generator/G_MODEL/A/Conv/weights/read:02?generator/G_MODEL/A/Conv/weights/Initializer/truncated_normal:08
ķ
$generator/G_MODEL/A/LayerNorm/beta:0)generator/G_MODEL/A/LayerNorm/beta/Assign)generator/G_MODEL/A/LayerNorm/beta/read:026generator/G_MODEL/A/LayerNorm/beta/Initializer/zeros:08
đ
%generator/G_MODEL/A/LayerNorm/gamma:0*generator/G_MODEL/A/LayerNorm/gamma/Assign*generator/G_MODEL/A/LayerNorm/gamma/read:026generator/G_MODEL/A/LayerNorm/gamma/Initializer/ones:08
Á
$generator/G_MODEL/A/Conv_1/weights:0)generator/G_MODEL/A/Conv_1/weights/Assign)generator/G_MODEL/A/Conv_1/weights/read:02Agenerator/G_MODEL/A/Conv_1/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/A/LayerNorm_1/beta:0+generator/G_MODEL/A/LayerNorm_1/beta/Assign+generator/G_MODEL/A/LayerNorm_1/beta/read:028generator/G_MODEL/A/LayerNorm_1/beta/Initializer/zeros:08
Á
'generator/G_MODEL/A/LayerNorm_1/gamma:0,generator/G_MODEL/A/LayerNorm_1/gamma/Assign,generator/G_MODEL/A/LayerNorm_1/gamma/read:028generator/G_MODEL/A/LayerNorm_1/gamma/Initializer/ones:08
Á
$generator/G_MODEL/A/Conv_2/weights:0)generator/G_MODEL/A/Conv_2/weights/Assign)generator/G_MODEL/A/Conv_2/weights/read:02Agenerator/G_MODEL/A/Conv_2/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/A/LayerNorm_2/beta:0+generator/G_MODEL/A/LayerNorm_2/beta/Assign+generator/G_MODEL/A/LayerNorm_2/beta/read:028generator/G_MODEL/A/LayerNorm_2/beta/Initializer/zeros:08
Á
'generator/G_MODEL/A/LayerNorm_2/gamma:0,generator/G_MODEL/A/LayerNorm_2/gamma/Assign,generator/G_MODEL/A/LayerNorm_2/gamma/read:028generator/G_MODEL/A/LayerNorm_2/gamma/Initializer/ones:08
đ
"generator/G_MODEL/B/Conv/weights:0'generator/G_MODEL/B/Conv/weights/Assign'generator/G_MODEL/B/Conv/weights/read:02?generator/G_MODEL/B/Conv/weights/Initializer/truncated_normal:08
ķ
$generator/G_MODEL/B/LayerNorm/beta:0)generator/G_MODEL/B/LayerNorm/beta/Assign)generator/G_MODEL/B/LayerNorm/beta/read:026generator/G_MODEL/B/LayerNorm/beta/Initializer/zeros:08
đ
%generator/G_MODEL/B/LayerNorm/gamma:0*generator/G_MODEL/B/LayerNorm/gamma/Assign*generator/G_MODEL/B/LayerNorm/gamma/read:026generator/G_MODEL/B/LayerNorm/gamma/Initializer/ones:08
Á
$generator/G_MODEL/B/Conv_1/weights:0)generator/G_MODEL/B/Conv_1/weights/Assign)generator/G_MODEL/B/Conv_1/weights/read:02Agenerator/G_MODEL/B/Conv_1/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/B/LayerNorm_1/beta:0+generator/G_MODEL/B/LayerNorm_1/beta/Assign+generator/G_MODEL/B/LayerNorm_1/beta/read:028generator/G_MODEL/B/LayerNorm_1/beta/Initializer/zeros:08
Á
'generator/G_MODEL/B/LayerNorm_1/gamma:0,generator/G_MODEL/B/LayerNorm_1/gamma/Assign,generator/G_MODEL/B/LayerNorm_1/gamma/read:028generator/G_MODEL/B/LayerNorm_1/gamma/Initializer/ones:08
đ
"generator/G_MODEL/C/Conv/weights:0'generator/G_MODEL/C/Conv/weights/Assign'generator/G_MODEL/C/Conv/weights/read:02?generator/G_MODEL/C/Conv/weights/Initializer/truncated_normal:08
ķ
$generator/G_MODEL/C/LayerNorm/beta:0)generator/G_MODEL/C/LayerNorm/beta/Assign)generator/G_MODEL/C/LayerNorm/beta/read:026generator/G_MODEL/C/LayerNorm/beta/Initializer/zeros:08
đ
%generator/G_MODEL/C/LayerNorm/gamma:0*generator/G_MODEL/C/LayerNorm/gamma/Assign*generator/G_MODEL/C/LayerNorm/gamma/read:026generator/G_MODEL/C/LayerNorm/gamma/Initializer/ones:08
Å
%generator/G_MODEL/C/r1/Conv/weights:0*generator/G_MODEL/C/r1/Conv/weights/Assign*generator/G_MODEL/C/r1/Conv/weights/read:02Bgenerator/G_MODEL/C/r1/Conv/weights/Initializer/truncated_normal:08
Â
'generator/G_MODEL/C/r1/LayerNorm/beta:0,generator/G_MODEL/C/r1/LayerNorm/beta/Assign,generator/G_MODEL/C/r1/LayerNorm/beta/read:029generator/G_MODEL/C/r1/LayerNorm/beta/Initializer/zeros:08
Å
(generator/G_MODEL/C/r1/LayerNorm/gamma:0-generator/G_MODEL/C/r1/LayerNorm/gamma/Assign-generator/G_MODEL/C/r1/LayerNorm/gamma/read:029generator/G_MODEL/C/r1/LayerNorm/gamma/Initializer/ones:08
Ĩ
generator/G_MODEL/C/r1/r1/w:0"generator/G_MODEL/C/r1/r1/w/Assign"generator/G_MODEL/C/r1/r1/w/read:02:generator/G_MODEL/C/r1/r1/w/Initializer/truncated_normal:08
Ķ
 generator/G_MODEL/C/r1/r1/bias:0%generator/G_MODEL/C/r1/r1/bias/Assign%generator/G_MODEL/C/r1/r1/bias/read:022generator/G_MODEL/C/r1/r1/bias/Initializer/Const:08
Ē
generator/G_MODEL/C/r1/1/beta:0$generator/G_MODEL/C/r1/1/beta/Assign$generator/G_MODEL/C/r1/1/beta/read:021generator/G_MODEL/C/r1/1/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r1/1/gamma:0%generator/G_MODEL/C/r1/1/gamma/Assign%generator/G_MODEL/C/r1/1/gamma/read:021generator/G_MODEL/C/r1/1/gamma/Initializer/ones:08
Í
'generator/G_MODEL/C/r1/Conv_1/weights:0,generator/G_MODEL/C/r1/Conv_1/weights/Assign,generator/G_MODEL/C/r1/Conv_1/weights/read:02Dgenerator/G_MODEL/C/r1/Conv_1/weights/Initializer/truncated_normal:08
Ē
generator/G_MODEL/C/r1/2/beta:0$generator/G_MODEL/C/r1/2/beta/Assign$generator/G_MODEL/C/r1/2/beta/read:021generator/G_MODEL/C/r1/2/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r1/2/gamma:0%generator/G_MODEL/C/r1/2/gamma/Assign%generator/G_MODEL/C/r1/2/gamma/read:021generator/G_MODEL/C/r1/2/gamma/Initializer/ones:08
Å
%generator/G_MODEL/C/r2/Conv/weights:0*generator/G_MODEL/C/r2/Conv/weights/Assign*generator/G_MODEL/C/r2/Conv/weights/read:02Bgenerator/G_MODEL/C/r2/Conv/weights/Initializer/truncated_normal:08
Â
'generator/G_MODEL/C/r2/LayerNorm/beta:0,generator/G_MODEL/C/r2/LayerNorm/beta/Assign,generator/G_MODEL/C/r2/LayerNorm/beta/read:029generator/G_MODEL/C/r2/LayerNorm/beta/Initializer/zeros:08
Å
(generator/G_MODEL/C/r2/LayerNorm/gamma:0-generator/G_MODEL/C/r2/LayerNorm/gamma/Assign-generator/G_MODEL/C/r2/LayerNorm/gamma/read:029generator/G_MODEL/C/r2/LayerNorm/gamma/Initializer/ones:08
Ĩ
generator/G_MODEL/C/r2/r2/w:0"generator/G_MODEL/C/r2/r2/w/Assign"generator/G_MODEL/C/r2/r2/w/read:02:generator/G_MODEL/C/r2/r2/w/Initializer/truncated_normal:08
Ķ
 generator/G_MODEL/C/r2/r2/bias:0%generator/G_MODEL/C/r2/r2/bias/Assign%generator/G_MODEL/C/r2/r2/bias/read:022generator/G_MODEL/C/r2/r2/bias/Initializer/Const:08
Ē
generator/G_MODEL/C/r2/1/beta:0$generator/G_MODEL/C/r2/1/beta/Assign$generator/G_MODEL/C/r2/1/beta/read:021generator/G_MODEL/C/r2/1/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r2/1/gamma:0%generator/G_MODEL/C/r2/1/gamma/Assign%generator/G_MODEL/C/r2/1/gamma/read:021generator/G_MODEL/C/r2/1/gamma/Initializer/ones:08
Í
'generator/G_MODEL/C/r2/Conv_1/weights:0,generator/G_MODEL/C/r2/Conv_1/weights/Assign,generator/G_MODEL/C/r2/Conv_1/weights/read:02Dgenerator/G_MODEL/C/r2/Conv_1/weights/Initializer/truncated_normal:08
Ē
generator/G_MODEL/C/r2/2/beta:0$generator/G_MODEL/C/r2/2/beta/Assign$generator/G_MODEL/C/r2/2/beta/read:021generator/G_MODEL/C/r2/2/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r2/2/gamma:0%generator/G_MODEL/C/r2/2/gamma/Assign%generator/G_MODEL/C/r2/2/gamma/read:021generator/G_MODEL/C/r2/2/gamma/Initializer/ones:08
Å
%generator/G_MODEL/C/r3/Conv/weights:0*generator/G_MODEL/C/r3/Conv/weights/Assign*generator/G_MODEL/C/r3/Conv/weights/read:02Bgenerator/G_MODEL/C/r3/Conv/weights/Initializer/truncated_normal:08
Â
'generator/G_MODEL/C/r3/LayerNorm/beta:0,generator/G_MODEL/C/r3/LayerNorm/beta/Assign,generator/G_MODEL/C/r3/LayerNorm/beta/read:029generator/G_MODEL/C/r3/LayerNorm/beta/Initializer/zeros:08
Å
(generator/G_MODEL/C/r3/LayerNorm/gamma:0-generator/G_MODEL/C/r3/LayerNorm/gamma/Assign-generator/G_MODEL/C/r3/LayerNorm/gamma/read:029generator/G_MODEL/C/r3/LayerNorm/gamma/Initializer/ones:08
Ĩ
generator/G_MODEL/C/r3/r3/w:0"generator/G_MODEL/C/r3/r3/w/Assign"generator/G_MODEL/C/r3/r3/w/read:02:generator/G_MODEL/C/r3/r3/w/Initializer/truncated_normal:08
Ķ
 generator/G_MODEL/C/r3/r3/bias:0%generator/G_MODEL/C/r3/r3/bias/Assign%generator/G_MODEL/C/r3/r3/bias/read:022generator/G_MODEL/C/r3/r3/bias/Initializer/Const:08
Ē
generator/G_MODEL/C/r3/1/beta:0$generator/G_MODEL/C/r3/1/beta/Assign$generator/G_MODEL/C/r3/1/beta/read:021generator/G_MODEL/C/r3/1/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r3/1/gamma:0%generator/G_MODEL/C/r3/1/gamma/Assign%generator/G_MODEL/C/r3/1/gamma/read:021generator/G_MODEL/C/r3/1/gamma/Initializer/ones:08
Í
'generator/G_MODEL/C/r3/Conv_1/weights:0,generator/G_MODEL/C/r3/Conv_1/weights/Assign,generator/G_MODEL/C/r3/Conv_1/weights/read:02Dgenerator/G_MODEL/C/r3/Conv_1/weights/Initializer/truncated_normal:08
Ē
generator/G_MODEL/C/r3/2/beta:0$generator/G_MODEL/C/r3/2/beta/Assign$generator/G_MODEL/C/r3/2/beta/read:021generator/G_MODEL/C/r3/2/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r3/2/gamma:0%generator/G_MODEL/C/r3/2/gamma/Assign%generator/G_MODEL/C/r3/2/gamma/read:021generator/G_MODEL/C/r3/2/gamma/Initializer/ones:08
Å
%generator/G_MODEL/C/r4/Conv/weights:0*generator/G_MODEL/C/r4/Conv/weights/Assign*generator/G_MODEL/C/r4/Conv/weights/read:02Bgenerator/G_MODEL/C/r4/Conv/weights/Initializer/truncated_normal:08
Â
'generator/G_MODEL/C/r4/LayerNorm/beta:0,generator/G_MODEL/C/r4/LayerNorm/beta/Assign,generator/G_MODEL/C/r4/LayerNorm/beta/read:029generator/G_MODEL/C/r4/LayerNorm/beta/Initializer/zeros:08
Å
(generator/G_MODEL/C/r4/LayerNorm/gamma:0-generator/G_MODEL/C/r4/LayerNorm/gamma/Assign-generator/G_MODEL/C/r4/LayerNorm/gamma/read:029generator/G_MODEL/C/r4/LayerNorm/gamma/Initializer/ones:08
Ĩ
generator/G_MODEL/C/r4/r4/w:0"generator/G_MODEL/C/r4/r4/w/Assign"generator/G_MODEL/C/r4/r4/w/read:02:generator/G_MODEL/C/r4/r4/w/Initializer/truncated_normal:08
Ķ
 generator/G_MODEL/C/r4/r4/bias:0%generator/G_MODEL/C/r4/r4/bias/Assign%generator/G_MODEL/C/r4/r4/bias/read:022generator/G_MODEL/C/r4/r4/bias/Initializer/Const:08
Ē
generator/G_MODEL/C/r4/1/beta:0$generator/G_MODEL/C/r4/1/beta/Assign$generator/G_MODEL/C/r4/1/beta/read:021generator/G_MODEL/C/r4/1/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r4/1/gamma:0%generator/G_MODEL/C/r4/1/gamma/Assign%generator/G_MODEL/C/r4/1/gamma/read:021generator/G_MODEL/C/r4/1/gamma/Initializer/ones:08
Í
'generator/G_MODEL/C/r4/Conv_1/weights:0,generator/G_MODEL/C/r4/Conv_1/weights/Assign,generator/G_MODEL/C/r4/Conv_1/weights/read:02Dgenerator/G_MODEL/C/r4/Conv_1/weights/Initializer/truncated_normal:08
Ē
generator/G_MODEL/C/r4/2/beta:0$generator/G_MODEL/C/r4/2/beta/Assign$generator/G_MODEL/C/r4/2/beta/read:021generator/G_MODEL/C/r4/2/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r4/2/gamma:0%generator/G_MODEL/C/r4/2/gamma/Assign%generator/G_MODEL/C/r4/2/gamma/read:021generator/G_MODEL/C/r4/2/gamma/Initializer/ones:08
Á
$generator/G_MODEL/C/Conv_1/weights:0)generator/G_MODEL/C/Conv_1/weights/Assign)generator/G_MODEL/C/Conv_1/weights/read:02Agenerator/G_MODEL/C/Conv_1/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/C/LayerNorm_1/beta:0+generator/G_MODEL/C/LayerNorm_1/beta/Assign+generator/G_MODEL/C/LayerNorm_1/beta/read:028generator/G_MODEL/C/LayerNorm_1/beta/Initializer/zeros:08
Á
'generator/G_MODEL/C/LayerNorm_1/gamma:0,generator/G_MODEL/C/LayerNorm_1/gamma/Assign,generator/G_MODEL/C/LayerNorm_1/gamma/read:028generator/G_MODEL/C/LayerNorm_1/gamma/Initializer/ones:08
đ
"generator/G_MODEL/D/Conv/weights:0'generator/G_MODEL/D/Conv/weights/Assign'generator/G_MODEL/D/Conv/weights/read:02?generator/G_MODEL/D/Conv/weights/Initializer/truncated_normal:08
ķ
$generator/G_MODEL/D/LayerNorm/beta:0)generator/G_MODEL/D/LayerNorm/beta/Assign)generator/G_MODEL/D/LayerNorm/beta/read:026generator/G_MODEL/D/LayerNorm/beta/Initializer/zeros:08
đ
%generator/G_MODEL/D/LayerNorm/gamma:0*generator/G_MODEL/D/LayerNorm/gamma/Assign*generator/G_MODEL/D/LayerNorm/gamma/read:026generator/G_MODEL/D/LayerNorm/gamma/Initializer/ones:08
Á
$generator/G_MODEL/D/Conv_1/weights:0)generator/G_MODEL/D/Conv_1/weights/Assign)generator/G_MODEL/D/Conv_1/weights/read:02Agenerator/G_MODEL/D/Conv_1/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/D/LayerNorm_1/beta:0+generator/G_MODEL/D/LayerNorm_1/beta/Assign+generator/G_MODEL/D/LayerNorm_1/beta/read:028generator/G_MODEL/D/LayerNorm_1/beta/Initializer/zeros:08
Á
'generator/G_MODEL/D/LayerNorm_1/gamma:0,generator/G_MODEL/D/LayerNorm_1/gamma/Assign,generator/G_MODEL/D/LayerNorm_1/gamma/read:028generator/G_MODEL/D/LayerNorm_1/gamma/Initializer/ones:08
đ
"generator/G_MODEL/E/Conv/weights:0'generator/G_MODEL/E/Conv/weights/Assign'generator/G_MODEL/E/Conv/weights/read:02?generator/G_MODEL/E/Conv/weights/Initializer/truncated_normal:08
ķ
$generator/G_MODEL/E/LayerNorm/beta:0)generator/G_MODEL/E/LayerNorm/beta/Assign)generator/G_MODEL/E/LayerNorm/beta/read:026generator/G_MODEL/E/LayerNorm/beta/Initializer/zeros:08
đ
%generator/G_MODEL/E/LayerNorm/gamma:0*generator/G_MODEL/E/LayerNorm/gamma/Assign*generator/G_MODEL/E/LayerNorm/gamma/read:026generator/G_MODEL/E/LayerNorm/gamma/Initializer/ones:08
Á
$generator/G_MODEL/E/Conv_1/weights:0)generator/G_MODEL/E/Conv_1/weights/Assign)generator/G_MODEL/E/Conv_1/weights/read:02Agenerator/G_MODEL/E/Conv_1/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/E/LayerNorm_1/beta:0+generator/G_MODEL/E/LayerNorm_1/beta/Assign+generator/G_MODEL/E/LayerNorm_1/beta/read:028generator/G_MODEL/E/LayerNorm_1/beta/Initializer/zeros:08
Á
'generator/G_MODEL/E/LayerNorm_1/gamma:0,generator/G_MODEL/E/LayerNorm_1/gamma/Assign,generator/G_MODEL/E/LayerNorm_1/gamma/read:028generator/G_MODEL/E/LayerNorm_1/gamma/Initializer/ones:08
Á
$generator/G_MODEL/E/Conv_2/weights:0)generator/G_MODEL/E/Conv_2/weights/Assign)generator/G_MODEL/E/Conv_2/weights/read:02Agenerator/G_MODEL/E/Conv_2/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/E/LayerNorm_2/beta:0+generator/G_MODEL/E/LayerNorm_2/beta/Assign+generator/G_MODEL/E/LayerNorm_2/beta/read:028generator/G_MODEL/E/LayerNorm_2/beta/Initializer/zeros:08
Á
'generator/G_MODEL/E/LayerNorm_2/gamma:0,generator/G_MODEL/E/LayerNorm_2/gamma/Assign,generator/G_MODEL/E/LayerNorm_2/gamma/read:028generator/G_MODEL/E/LayerNorm_2/gamma/Initializer/ones:08
Ų
*generator/G_MODEL/out_layer/Conv/weights:0/generator/G_MODEL/out_layer/Conv/weights/Assign/generator/G_MODEL/out_layer/Conv/weights/read:02Ggenerator/G_MODEL/out_layer/Conv/weights/Initializer/truncated_normal:08"ãe
model_variablesÏeĖe
đ
"generator/G_MODEL/A/Conv/weights:0'generator/G_MODEL/A/Conv/weights/Assign'generator/G_MODEL/A/Conv/weights/read:02?generator/G_MODEL/A/Conv/weights/Initializer/truncated_normal:08
ķ
$generator/G_MODEL/A/LayerNorm/beta:0)generator/G_MODEL/A/LayerNorm/beta/Assign)generator/G_MODEL/A/LayerNorm/beta/read:026generator/G_MODEL/A/LayerNorm/beta/Initializer/zeros:08
đ
%generator/G_MODEL/A/LayerNorm/gamma:0*generator/G_MODEL/A/LayerNorm/gamma/Assign*generator/G_MODEL/A/LayerNorm/gamma/read:026generator/G_MODEL/A/LayerNorm/gamma/Initializer/ones:08
Á
$generator/G_MODEL/A/Conv_1/weights:0)generator/G_MODEL/A/Conv_1/weights/Assign)generator/G_MODEL/A/Conv_1/weights/read:02Agenerator/G_MODEL/A/Conv_1/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/A/LayerNorm_1/beta:0+generator/G_MODEL/A/LayerNorm_1/beta/Assign+generator/G_MODEL/A/LayerNorm_1/beta/read:028generator/G_MODEL/A/LayerNorm_1/beta/Initializer/zeros:08
Á
'generator/G_MODEL/A/LayerNorm_1/gamma:0,generator/G_MODEL/A/LayerNorm_1/gamma/Assign,generator/G_MODEL/A/LayerNorm_1/gamma/read:028generator/G_MODEL/A/LayerNorm_1/gamma/Initializer/ones:08
Á
$generator/G_MODEL/A/Conv_2/weights:0)generator/G_MODEL/A/Conv_2/weights/Assign)generator/G_MODEL/A/Conv_2/weights/read:02Agenerator/G_MODEL/A/Conv_2/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/A/LayerNorm_2/beta:0+generator/G_MODEL/A/LayerNorm_2/beta/Assign+generator/G_MODEL/A/LayerNorm_2/beta/read:028generator/G_MODEL/A/LayerNorm_2/beta/Initializer/zeros:08
Á
'generator/G_MODEL/A/LayerNorm_2/gamma:0,generator/G_MODEL/A/LayerNorm_2/gamma/Assign,generator/G_MODEL/A/LayerNorm_2/gamma/read:028generator/G_MODEL/A/LayerNorm_2/gamma/Initializer/ones:08
đ
"generator/G_MODEL/B/Conv/weights:0'generator/G_MODEL/B/Conv/weights/Assign'generator/G_MODEL/B/Conv/weights/read:02?generator/G_MODEL/B/Conv/weights/Initializer/truncated_normal:08
ķ
$generator/G_MODEL/B/LayerNorm/beta:0)generator/G_MODEL/B/LayerNorm/beta/Assign)generator/G_MODEL/B/LayerNorm/beta/read:026generator/G_MODEL/B/LayerNorm/beta/Initializer/zeros:08
đ
%generator/G_MODEL/B/LayerNorm/gamma:0*generator/G_MODEL/B/LayerNorm/gamma/Assign*generator/G_MODEL/B/LayerNorm/gamma/read:026generator/G_MODEL/B/LayerNorm/gamma/Initializer/ones:08
Á
$generator/G_MODEL/B/Conv_1/weights:0)generator/G_MODEL/B/Conv_1/weights/Assign)generator/G_MODEL/B/Conv_1/weights/read:02Agenerator/G_MODEL/B/Conv_1/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/B/LayerNorm_1/beta:0+generator/G_MODEL/B/LayerNorm_1/beta/Assign+generator/G_MODEL/B/LayerNorm_1/beta/read:028generator/G_MODEL/B/LayerNorm_1/beta/Initializer/zeros:08
Á
'generator/G_MODEL/B/LayerNorm_1/gamma:0,generator/G_MODEL/B/LayerNorm_1/gamma/Assign,generator/G_MODEL/B/LayerNorm_1/gamma/read:028generator/G_MODEL/B/LayerNorm_1/gamma/Initializer/ones:08
đ
"generator/G_MODEL/C/Conv/weights:0'generator/G_MODEL/C/Conv/weights/Assign'generator/G_MODEL/C/Conv/weights/read:02?generator/G_MODEL/C/Conv/weights/Initializer/truncated_normal:08
ķ
$generator/G_MODEL/C/LayerNorm/beta:0)generator/G_MODEL/C/LayerNorm/beta/Assign)generator/G_MODEL/C/LayerNorm/beta/read:026generator/G_MODEL/C/LayerNorm/beta/Initializer/zeros:08
đ
%generator/G_MODEL/C/LayerNorm/gamma:0*generator/G_MODEL/C/LayerNorm/gamma/Assign*generator/G_MODEL/C/LayerNorm/gamma/read:026generator/G_MODEL/C/LayerNorm/gamma/Initializer/ones:08
Å
%generator/G_MODEL/C/r1/Conv/weights:0*generator/G_MODEL/C/r1/Conv/weights/Assign*generator/G_MODEL/C/r1/Conv/weights/read:02Bgenerator/G_MODEL/C/r1/Conv/weights/Initializer/truncated_normal:08
Â
'generator/G_MODEL/C/r1/LayerNorm/beta:0,generator/G_MODEL/C/r1/LayerNorm/beta/Assign,generator/G_MODEL/C/r1/LayerNorm/beta/read:029generator/G_MODEL/C/r1/LayerNorm/beta/Initializer/zeros:08
Å
(generator/G_MODEL/C/r1/LayerNorm/gamma:0-generator/G_MODEL/C/r1/LayerNorm/gamma/Assign-generator/G_MODEL/C/r1/LayerNorm/gamma/read:029generator/G_MODEL/C/r1/LayerNorm/gamma/Initializer/ones:08
Ē
generator/G_MODEL/C/r1/1/beta:0$generator/G_MODEL/C/r1/1/beta/Assign$generator/G_MODEL/C/r1/1/beta/read:021generator/G_MODEL/C/r1/1/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r1/1/gamma:0%generator/G_MODEL/C/r1/1/gamma/Assign%generator/G_MODEL/C/r1/1/gamma/read:021generator/G_MODEL/C/r1/1/gamma/Initializer/ones:08
Í
'generator/G_MODEL/C/r1/Conv_1/weights:0,generator/G_MODEL/C/r1/Conv_1/weights/Assign,generator/G_MODEL/C/r1/Conv_1/weights/read:02Dgenerator/G_MODEL/C/r1/Conv_1/weights/Initializer/truncated_normal:08
Ē
generator/G_MODEL/C/r1/2/beta:0$generator/G_MODEL/C/r1/2/beta/Assign$generator/G_MODEL/C/r1/2/beta/read:021generator/G_MODEL/C/r1/2/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r1/2/gamma:0%generator/G_MODEL/C/r1/2/gamma/Assign%generator/G_MODEL/C/r1/2/gamma/read:021generator/G_MODEL/C/r1/2/gamma/Initializer/ones:08
Å
%generator/G_MODEL/C/r2/Conv/weights:0*generator/G_MODEL/C/r2/Conv/weights/Assign*generator/G_MODEL/C/r2/Conv/weights/read:02Bgenerator/G_MODEL/C/r2/Conv/weights/Initializer/truncated_normal:08
Â
'generator/G_MODEL/C/r2/LayerNorm/beta:0,generator/G_MODEL/C/r2/LayerNorm/beta/Assign,generator/G_MODEL/C/r2/LayerNorm/beta/read:029generator/G_MODEL/C/r2/LayerNorm/beta/Initializer/zeros:08
Å
(generator/G_MODEL/C/r2/LayerNorm/gamma:0-generator/G_MODEL/C/r2/LayerNorm/gamma/Assign-generator/G_MODEL/C/r2/LayerNorm/gamma/read:029generator/G_MODEL/C/r2/LayerNorm/gamma/Initializer/ones:08
Ē
generator/G_MODEL/C/r2/1/beta:0$generator/G_MODEL/C/r2/1/beta/Assign$generator/G_MODEL/C/r2/1/beta/read:021generator/G_MODEL/C/r2/1/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r2/1/gamma:0%generator/G_MODEL/C/r2/1/gamma/Assign%generator/G_MODEL/C/r2/1/gamma/read:021generator/G_MODEL/C/r2/1/gamma/Initializer/ones:08
Í
'generator/G_MODEL/C/r2/Conv_1/weights:0,generator/G_MODEL/C/r2/Conv_1/weights/Assign,generator/G_MODEL/C/r2/Conv_1/weights/read:02Dgenerator/G_MODEL/C/r2/Conv_1/weights/Initializer/truncated_normal:08
Ē
generator/G_MODEL/C/r2/2/beta:0$generator/G_MODEL/C/r2/2/beta/Assign$generator/G_MODEL/C/r2/2/beta/read:021generator/G_MODEL/C/r2/2/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r2/2/gamma:0%generator/G_MODEL/C/r2/2/gamma/Assign%generator/G_MODEL/C/r2/2/gamma/read:021generator/G_MODEL/C/r2/2/gamma/Initializer/ones:08
Å
%generator/G_MODEL/C/r3/Conv/weights:0*generator/G_MODEL/C/r3/Conv/weights/Assign*generator/G_MODEL/C/r3/Conv/weights/read:02Bgenerator/G_MODEL/C/r3/Conv/weights/Initializer/truncated_normal:08
Â
'generator/G_MODEL/C/r3/LayerNorm/beta:0,generator/G_MODEL/C/r3/LayerNorm/beta/Assign,generator/G_MODEL/C/r3/LayerNorm/beta/read:029generator/G_MODEL/C/r3/LayerNorm/beta/Initializer/zeros:08
Å
(generator/G_MODEL/C/r3/LayerNorm/gamma:0-generator/G_MODEL/C/r3/LayerNorm/gamma/Assign-generator/G_MODEL/C/r3/LayerNorm/gamma/read:029generator/G_MODEL/C/r3/LayerNorm/gamma/Initializer/ones:08
Ē
generator/G_MODEL/C/r3/1/beta:0$generator/G_MODEL/C/r3/1/beta/Assign$generator/G_MODEL/C/r3/1/beta/read:021generator/G_MODEL/C/r3/1/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r3/1/gamma:0%generator/G_MODEL/C/r3/1/gamma/Assign%generator/G_MODEL/C/r3/1/gamma/read:021generator/G_MODEL/C/r3/1/gamma/Initializer/ones:08
Í
'generator/G_MODEL/C/r3/Conv_1/weights:0,generator/G_MODEL/C/r3/Conv_1/weights/Assign,generator/G_MODEL/C/r3/Conv_1/weights/read:02Dgenerator/G_MODEL/C/r3/Conv_1/weights/Initializer/truncated_normal:08
Ē
generator/G_MODEL/C/r3/2/beta:0$generator/G_MODEL/C/r3/2/beta/Assign$generator/G_MODEL/C/r3/2/beta/read:021generator/G_MODEL/C/r3/2/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r3/2/gamma:0%generator/G_MODEL/C/r3/2/gamma/Assign%generator/G_MODEL/C/r3/2/gamma/read:021generator/G_MODEL/C/r3/2/gamma/Initializer/ones:08
Å
%generator/G_MODEL/C/r4/Conv/weights:0*generator/G_MODEL/C/r4/Conv/weights/Assign*generator/G_MODEL/C/r4/Conv/weights/read:02Bgenerator/G_MODEL/C/r4/Conv/weights/Initializer/truncated_normal:08
Â
'generator/G_MODEL/C/r4/LayerNorm/beta:0,generator/G_MODEL/C/r4/LayerNorm/beta/Assign,generator/G_MODEL/C/r4/LayerNorm/beta/read:029generator/G_MODEL/C/r4/LayerNorm/beta/Initializer/zeros:08
Å
(generator/G_MODEL/C/r4/LayerNorm/gamma:0-generator/G_MODEL/C/r4/LayerNorm/gamma/Assign-generator/G_MODEL/C/r4/LayerNorm/gamma/read:029generator/G_MODEL/C/r4/LayerNorm/gamma/Initializer/ones:08
Ē
generator/G_MODEL/C/r4/1/beta:0$generator/G_MODEL/C/r4/1/beta/Assign$generator/G_MODEL/C/r4/1/beta/read:021generator/G_MODEL/C/r4/1/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r4/1/gamma:0%generator/G_MODEL/C/r4/1/gamma/Assign%generator/G_MODEL/C/r4/1/gamma/read:021generator/G_MODEL/C/r4/1/gamma/Initializer/ones:08
Í
'generator/G_MODEL/C/r4/Conv_1/weights:0,generator/G_MODEL/C/r4/Conv_1/weights/Assign,generator/G_MODEL/C/r4/Conv_1/weights/read:02Dgenerator/G_MODEL/C/r4/Conv_1/weights/Initializer/truncated_normal:08
Ē
generator/G_MODEL/C/r4/2/beta:0$generator/G_MODEL/C/r4/2/beta/Assign$generator/G_MODEL/C/r4/2/beta/read:021generator/G_MODEL/C/r4/2/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r4/2/gamma:0%generator/G_MODEL/C/r4/2/gamma/Assign%generator/G_MODEL/C/r4/2/gamma/read:021generator/G_MODEL/C/r4/2/gamma/Initializer/ones:08
Á
$generator/G_MODEL/C/Conv_1/weights:0)generator/G_MODEL/C/Conv_1/weights/Assign)generator/G_MODEL/C/Conv_1/weights/read:02Agenerator/G_MODEL/C/Conv_1/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/C/LayerNorm_1/beta:0+generator/G_MODEL/C/LayerNorm_1/beta/Assign+generator/G_MODEL/C/LayerNorm_1/beta/read:028generator/G_MODEL/C/LayerNorm_1/beta/Initializer/zeros:08
Á
'generator/G_MODEL/C/LayerNorm_1/gamma:0,generator/G_MODEL/C/LayerNorm_1/gamma/Assign,generator/G_MODEL/C/LayerNorm_1/gamma/read:028generator/G_MODEL/C/LayerNorm_1/gamma/Initializer/ones:08
đ
"generator/G_MODEL/D/Conv/weights:0'generator/G_MODEL/D/Conv/weights/Assign'generator/G_MODEL/D/Conv/weights/read:02?generator/G_MODEL/D/Conv/weights/Initializer/truncated_normal:08
ķ
$generator/G_MODEL/D/LayerNorm/beta:0)generator/G_MODEL/D/LayerNorm/beta/Assign)generator/G_MODEL/D/LayerNorm/beta/read:026generator/G_MODEL/D/LayerNorm/beta/Initializer/zeros:08
đ
%generator/G_MODEL/D/LayerNorm/gamma:0*generator/G_MODEL/D/LayerNorm/gamma/Assign*generator/G_MODEL/D/LayerNorm/gamma/read:026generator/G_MODEL/D/LayerNorm/gamma/Initializer/ones:08
Á
$generator/G_MODEL/D/Conv_1/weights:0)generator/G_MODEL/D/Conv_1/weights/Assign)generator/G_MODEL/D/Conv_1/weights/read:02Agenerator/G_MODEL/D/Conv_1/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/D/LayerNorm_1/beta:0+generator/G_MODEL/D/LayerNorm_1/beta/Assign+generator/G_MODEL/D/LayerNorm_1/beta/read:028generator/G_MODEL/D/LayerNorm_1/beta/Initializer/zeros:08
Á
'generator/G_MODEL/D/LayerNorm_1/gamma:0,generator/G_MODEL/D/LayerNorm_1/gamma/Assign,generator/G_MODEL/D/LayerNorm_1/gamma/read:028generator/G_MODEL/D/LayerNorm_1/gamma/Initializer/ones:08
đ
"generator/G_MODEL/E/Conv/weights:0'generator/G_MODEL/E/Conv/weights/Assign'generator/G_MODEL/E/Conv/weights/read:02?generator/G_MODEL/E/Conv/weights/Initializer/truncated_normal:08
ķ
$generator/G_MODEL/E/LayerNorm/beta:0)generator/G_MODEL/E/LayerNorm/beta/Assign)generator/G_MODEL/E/LayerNorm/beta/read:026generator/G_MODEL/E/LayerNorm/beta/Initializer/zeros:08
đ
%generator/G_MODEL/E/LayerNorm/gamma:0*generator/G_MODEL/E/LayerNorm/gamma/Assign*generator/G_MODEL/E/LayerNorm/gamma/read:026generator/G_MODEL/E/LayerNorm/gamma/Initializer/ones:08
Á
$generator/G_MODEL/E/Conv_1/weights:0)generator/G_MODEL/E/Conv_1/weights/Assign)generator/G_MODEL/E/Conv_1/weights/read:02Agenerator/G_MODEL/E/Conv_1/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/E/LayerNorm_1/beta:0+generator/G_MODEL/E/LayerNorm_1/beta/Assign+generator/G_MODEL/E/LayerNorm_1/beta/read:028generator/G_MODEL/E/LayerNorm_1/beta/Initializer/zeros:08
Á
'generator/G_MODEL/E/LayerNorm_1/gamma:0,generator/G_MODEL/E/LayerNorm_1/gamma/Assign,generator/G_MODEL/E/LayerNorm_1/gamma/read:028generator/G_MODEL/E/LayerNorm_1/gamma/Initializer/ones:08
Á
$generator/G_MODEL/E/Conv_2/weights:0)generator/G_MODEL/E/Conv_2/weights/Assign)generator/G_MODEL/E/Conv_2/weights/read:02Agenerator/G_MODEL/E/Conv_2/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/E/LayerNorm_2/beta:0+generator/G_MODEL/E/LayerNorm_2/beta/Assign+generator/G_MODEL/E/LayerNorm_2/beta/read:028generator/G_MODEL/E/LayerNorm_2/beta/Initializer/zeros:08
Á
'generator/G_MODEL/E/LayerNorm_2/gamma:0,generator/G_MODEL/E/LayerNorm_2/gamma/Assign,generator/G_MODEL/E/LayerNorm_2/gamma/read:028generator/G_MODEL/E/LayerNorm_2/gamma/Initializer/ones:08
Ų
*generator/G_MODEL/out_layer/Conv/weights:0/generator/G_MODEL/out_layer/Conv/weights/Assign/generator/G_MODEL/out_layer/Conv/weights/read:02Ggenerator/G_MODEL/out_layer/Conv/weights/Initializer/truncated_normal:08"Ŧp
trainable_variablespp
đ
"generator/G_MODEL/A/Conv/weights:0'generator/G_MODEL/A/Conv/weights/Assign'generator/G_MODEL/A/Conv/weights/read:02?generator/G_MODEL/A/Conv/weights/Initializer/truncated_normal:08
ķ
$generator/G_MODEL/A/LayerNorm/beta:0)generator/G_MODEL/A/LayerNorm/beta/Assign)generator/G_MODEL/A/LayerNorm/beta/read:026generator/G_MODEL/A/LayerNorm/beta/Initializer/zeros:08
đ
%generator/G_MODEL/A/LayerNorm/gamma:0*generator/G_MODEL/A/LayerNorm/gamma/Assign*generator/G_MODEL/A/LayerNorm/gamma/read:026generator/G_MODEL/A/LayerNorm/gamma/Initializer/ones:08
Á
$generator/G_MODEL/A/Conv_1/weights:0)generator/G_MODEL/A/Conv_1/weights/Assign)generator/G_MODEL/A/Conv_1/weights/read:02Agenerator/G_MODEL/A/Conv_1/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/A/LayerNorm_1/beta:0+generator/G_MODEL/A/LayerNorm_1/beta/Assign+generator/G_MODEL/A/LayerNorm_1/beta/read:028generator/G_MODEL/A/LayerNorm_1/beta/Initializer/zeros:08
Á
'generator/G_MODEL/A/LayerNorm_1/gamma:0,generator/G_MODEL/A/LayerNorm_1/gamma/Assign,generator/G_MODEL/A/LayerNorm_1/gamma/read:028generator/G_MODEL/A/LayerNorm_1/gamma/Initializer/ones:08
Á
$generator/G_MODEL/A/Conv_2/weights:0)generator/G_MODEL/A/Conv_2/weights/Assign)generator/G_MODEL/A/Conv_2/weights/read:02Agenerator/G_MODEL/A/Conv_2/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/A/LayerNorm_2/beta:0+generator/G_MODEL/A/LayerNorm_2/beta/Assign+generator/G_MODEL/A/LayerNorm_2/beta/read:028generator/G_MODEL/A/LayerNorm_2/beta/Initializer/zeros:08
Á
'generator/G_MODEL/A/LayerNorm_2/gamma:0,generator/G_MODEL/A/LayerNorm_2/gamma/Assign,generator/G_MODEL/A/LayerNorm_2/gamma/read:028generator/G_MODEL/A/LayerNorm_2/gamma/Initializer/ones:08
đ
"generator/G_MODEL/B/Conv/weights:0'generator/G_MODEL/B/Conv/weights/Assign'generator/G_MODEL/B/Conv/weights/read:02?generator/G_MODEL/B/Conv/weights/Initializer/truncated_normal:08
ķ
$generator/G_MODEL/B/LayerNorm/beta:0)generator/G_MODEL/B/LayerNorm/beta/Assign)generator/G_MODEL/B/LayerNorm/beta/read:026generator/G_MODEL/B/LayerNorm/beta/Initializer/zeros:08
đ
%generator/G_MODEL/B/LayerNorm/gamma:0*generator/G_MODEL/B/LayerNorm/gamma/Assign*generator/G_MODEL/B/LayerNorm/gamma/read:026generator/G_MODEL/B/LayerNorm/gamma/Initializer/ones:08
Á
$generator/G_MODEL/B/Conv_1/weights:0)generator/G_MODEL/B/Conv_1/weights/Assign)generator/G_MODEL/B/Conv_1/weights/read:02Agenerator/G_MODEL/B/Conv_1/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/B/LayerNorm_1/beta:0+generator/G_MODEL/B/LayerNorm_1/beta/Assign+generator/G_MODEL/B/LayerNorm_1/beta/read:028generator/G_MODEL/B/LayerNorm_1/beta/Initializer/zeros:08
Á
'generator/G_MODEL/B/LayerNorm_1/gamma:0,generator/G_MODEL/B/LayerNorm_1/gamma/Assign,generator/G_MODEL/B/LayerNorm_1/gamma/read:028generator/G_MODEL/B/LayerNorm_1/gamma/Initializer/ones:08
đ
"generator/G_MODEL/C/Conv/weights:0'generator/G_MODEL/C/Conv/weights/Assign'generator/G_MODEL/C/Conv/weights/read:02?generator/G_MODEL/C/Conv/weights/Initializer/truncated_normal:08
ķ
$generator/G_MODEL/C/LayerNorm/beta:0)generator/G_MODEL/C/LayerNorm/beta/Assign)generator/G_MODEL/C/LayerNorm/beta/read:026generator/G_MODEL/C/LayerNorm/beta/Initializer/zeros:08
đ
%generator/G_MODEL/C/LayerNorm/gamma:0*generator/G_MODEL/C/LayerNorm/gamma/Assign*generator/G_MODEL/C/LayerNorm/gamma/read:026generator/G_MODEL/C/LayerNorm/gamma/Initializer/ones:08
Å
%generator/G_MODEL/C/r1/Conv/weights:0*generator/G_MODEL/C/r1/Conv/weights/Assign*generator/G_MODEL/C/r1/Conv/weights/read:02Bgenerator/G_MODEL/C/r1/Conv/weights/Initializer/truncated_normal:08
Â
'generator/G_MODEL/C/r1/LayerNorm/beta:0,generator/G_MODEL/C/r1/LayerNorm/beta/Assign,generator/G_MODEL/C/r1/LayerNorm/beta/read:029generator/G_MODEL/C/r1/LayerNorm/beta/Initializer/zeros:08
Å
(generator/G_MODEL/C/r1/LayerNorm/gamma:0-generator/G_MODEL/C/r1/LayerNorm/gamma/Assign-generator/G_MODEL/C/r1/LayerNorm/gamma/read:029generator/G_MODEL/C/r1/LayerNorm/gamma/Initializer/ones:08
Ĩ
generator/G_MODEL/C/r1/r1/w:0"generator/G_MODEL/C/r1/r1/w/Assign"generator/G_MODEL/C/r1/r1/w/read:02:generator/G_MODEL/C/r1/r1/w/Initializer/truncated_normal:08
Ķ
 generator/G_MODEL/C/r1/r1/bias:0%generator/G_MODEL/C/r1/r1/bias/Assign%generator/G_MODEL/C/r1/r1/bias/read:022generator/G_MODEL/C/r1/r1/bias/Initializer/Const:08
Ē
generator/G_MODEL/C/r1/1/beta:0$generator/G_MODEL/C/r1/1/beta/Assign$generator/G_MODEL/C/r1/1/beta/read:021generator/G_MODEL/C/r1/1/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r1/1/gamma:0%generator/G_MODEL/C/r1/1/gamma/Assign%generator/G_MODEL/C/r1/1/gamma/read:021generator/G_MODEL/C/r1/1/gamma/Initializer/ones:08
Í
'generator/G_MODEL/C/r1/Conv_1/weights:0,generator/G_MODEL/C/r1/Conv_1/weights/Assign,generator/G_MODEL/C/r1/Conv_1/weights/read:02Dgenerator/G_MODEL/C/r1/Conv_1/weights/Initializer/truncated_normal:08
Ē
generator/G_MODEL/C/r1/2/beta:0$generator/G_MODEL/C/r1/2/beta/Assign$generator/G_MODEL/C/r1/2/beta/read:021generator/G_MODEL/C/r1/2/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r1/2/gamma:0%generator/G_MODEL/C/r1/2/gamma/Assign%generator/G_MODEL/C/r1/2/gamma/read:021generator/G_MODEL/C/r1/2/gamma/Initializer/ones:08
Å
%generator/G_MODEL/C/r2/Conv/weights:0*generator/G_MODEL/C/r2/Conv/weights/Assign*generator/G_MODEL/C/r2/Conv/weights/read:02Bgenerator/G_MODEL/C/r2/Conv/weights/Initializer/truncated_normal:08
Â
'generator/G_MODEL/C/r2/LayerNorm/beta:0,generator/G_MODEL/C/r2/LayerNorm/beta/Assign,generator/G_MODEL/C/r2/LayerNorm/beta/read:029generator/G_MODEL/C/r2/LayerNorm/beta/Initializer/zeros:08
Å
(generator/G_MODEL/C/r2/LayerNorm/gamma:0-generator/G_MODEL/C/r2/LayerNorm/gamma/Assign-generator/G_MODEL/C/r2/LayerNorm/gamma/read:029generator/G_MODEL/C/r2/LayerNorm/gamma/Initializer/ones:08
Ĩ
generator/G_MODEL/C/r2/r2/w:0"generator/G_MODEL/C/r2/r2/w/Assign"generator/G_MODEL/C/r2/r2/w/read:02:generator/G_MODEL/C/r2/r2/w/Initializer/truncated_normal:08
Ķ
 generator/G_MODEL/C/r2/r2/bias:0%generator/G_MODEL/C/r2/r2/bias/Assign%generator/G_MODEL/C/r2/r2/bias/read:022generator/G_MODEL/C/r2/r2/bias/Initializer/Const:08
Ē
generator/G_MODEL/C/r2/1/beta:0$generator/G_MODEL/C/r2/1/beta/Assign$generator/G_MODEL/C/r2/1/beta/read:021generator/G_MODEL/C/r2/1/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r2/1/gamma:0%generator/G_MODEL/C/r2/1/gamma/Assign%generator/G_MODEL/C/r2/1/gamma/read:021generator/G_MODEL/C/r2/1/gamma/Initializer/ones:08
Í
'generator/G_MODEL/C/r2/Conv_1/weights:0,generator/G_MODEL/C/r2/Conv_1/weights/Assign,generator/G_MODEL/C/r2/Conv_1/weights/read:02Dgenerator/G_MODEL/C/r2/Conv_1/weights/Initializer/truncated_normal:08
Ē
generator/G_MODEL/C/r2/2/beta:0$generator/G_MODEL/C/r2/2/beta/Assign$generator/G_MODEL/C/r2/2/beta/read:021generator/G_MODEL/C/r2/2/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r2/2/gamma:0%generator/G_MODEL/C/r2/2/gamma/Assign%generator/G_MODEL/C/r2/2/gamma/read:021generator/G_MODEL/C/r2/2/gamma/Initializer/ones:08
Å
%generator/G_MODEL/C/r3/Conv/weights:0*generator/G_MODEL/C/r3/Conv/weights/Assign*generator/G_MODEL/C/r3/Conv/weights/read:02Bgenerator/G_MODEL/C/r3/Conv/weights/Initializer/truncated_normal:08
Â
'generator/G_MODEL/C/r3/LayerNorm/beta:0,generator/G_MODEL/C/r3/LayerNorm/beta/Assign,generator/G_MODEL/C/r3/LayerNorm/beta/read:029generator/G_MODEL/C/r3/LayerNorm/beta/Initializer/zeros:08
Å
(generator/G_MODEL/C/r3/LayerNorm/gamma:0-generator/G_MODEL/C/r3/LayerNorm/gamma/Assign-generator/G_MODEL/C/r3/LayerNorm/gamma/read:029generator/G_MODEL/C/r3/LayerNorm/gamma/Initializer/ones:08
Ĩ
generator/G_MODEL/C/r3/r3/w:0"generator/G_MODEL/C/r3/r3/w/Assign"generator/G_MODEL/C/r3/r3/w/read:02:generator/G_MODEL/C/r3/r3/w/Initializer/truncated_normal:08
Ķ
 generator/G_MODEL/C/r3/r3/bias:0%generator/G_MODEL/C/r3/r3/bias/Assign%generator/G_MODEL/C/r3/r3/bias/read:022generator/G_MODEL/C/r3/r3/bias/Initializer/Const:08
Ē
generator/G_MODEL/C/r3/1/beta:0$generator/G_MODEL/C/r3/1/beta/Assign$generator/G_MODEL/C/r3/1/beta/read:021generator/G_MODEL/C/r3/1/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r3/1/gamma:0%generator/G_MODEL/C/r3/1/gamma/Assign%generator/G_MODEL/C/r3/1/gamma/read:021generator/G_MODEL/C/r3/1/gamma/Initializer/ones:08
Í
'generator/G_MODEL/C/r3/Conv_1/weights:0,generator/G_MODEL/C/r3/Conv_1/weights/Assign,generator/G_MODEL/C/r3/Conv_1/weights/read:02Dgenerator/G_MODEL/C/r3/Conv_1/weights/Initializer/truncated_normal:08
Ē
generator/G_MODEL/C/r3/2/beta:0$generator/G_MODEL/C/r3/2/beta/Assign$generator/G_MODEL/C/r3/2/beta/read:021generator/G_MODEL/C/r3/2/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r3/2/gamma:0%generator/G_MODEL/C/r3/2/gamma/Assign%generator/G_MODEL/C/r3/2/gamma/read:021generator/G_MODEL/C/r3/2/gamma/Initializer/ones:08
Å
%generator/G_MODEL/C/r4/Conv/weights:0*generator/G_MODEL/C/r4/Conv/weights/Assign*generator/G_MODEL/C/r4/Conv/weights/read:02Bgenerator/G_MODEL/C/r4/Conv/weights/Initializer/truncated_normal:08
Â
'generator/G_MODEL/C/r4/LayerNorm/beta:0,generator/G_MODEL/C/r4/LayerNorm/beta/Assign,generator/G_MODEL/C/r4/LayerNorm/beta/read:029generator/G_MODEL/C/r4/LayerNorm/beta/Initializer/zeros:08
Å
(generator/G_MODEL/C/r4/LayerNorm/gamma:0-generator/G_MODEL/C/r4/LayerNorm/gamma/Assign-generator/G_MODEL/C/r4/LayerNorm/gamma/read:029generator/G_MODEL/C/r4/LayerNorm/gamma/Initializer/ones:08
Ĩ
generator/G_MODEL/C/r4/r4/w:0"generator/G_MODEL/C/r4/r4/w/Assign"generator/G_MODEL/C/r4/r4/w/read:02:generator/G_MODEL/C/r4/r4/w/Initializer/truncated_normal:08
Ķ
 generator/G_MODEL/C/r4/r4/bias:0%generator/G_MODEL/C/r4/r4/bias/Assign%generator/G_MODEL/C/r4/r4/bias/read:022generator/G_MODEL/C/r4/r4/bias/Initializer/Const:08
Ē
generator/G_MODEL/C/r4/1/beta:0$generator/G_MODEL/C/r4/1/beta/Assign$generator/G_MODEL/C/r4/1/beta/read:021generator/G_MODEL/C/r4/1/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r4/1/gamma:0%generator/G_MODEL/C/r4/1/gamma/Assign%generator/G_MODEL/C/r4/1/gamma/read:021generator/G_MODEL/C/r4/1/gamma/Initializer/ones:08
Í
'generator/G_MODEL/C/r4/Conv_1/weights:0,generator/G_MODEL/C/r4/Conv_1/weights/Assign,generator/G_MODEL/C/r4/Conv_1/weights/read:02Dgenerator/G_MODEL/C/r4/Conv_1/weights/Initializer/truncated_normal:08
Ē
generator/G_MODEL/C/r4/2/beta:0$generator/G_MODEL/C/r4/2/beta/Assign$generator/G_MODEL/C/r4/2/beta/read:021generator/G_MODEL/C/r4/2/beta/Initializer/zeros:08
Ĩ
 generator/G_MODEL/C/r4/2/gamma:0%generator/G_MODEL/C/r4/2/gamma/Assign%generator/G_MODEL/C/r4/2/gamma/read:021generator/G_MODEL/C/r4/2/gamma/Initializer/ones:08
Á
$generator/G_MODEL/C/Conv_1/weights:0)generator/G_MODEL/C/Conv_1/weights/Assign)generator/G_MODEL/C/Conv_1/weights/read:02Agenerator/G_MODEL/C/Conv_1/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/C/LayerNorm_1/beta:0+generator/G_MODEL/C/LayerNorm_1/beta/Assign+generator/G_MODEL/C/LayerNorm_1/beta/read:028generator/G_MODEL/C/LayerNorm_1/beta/Initializer/zeros:08
Á
'generator/G_MODEL/C/LayerNorm_1/gamma:0,generator/G_MODEL/C/LayerNorm_1/gamma/Assign,generator/G_MODEL/C/LayerNorm_1/gamma/read:028generator/G_MODEL/C/LayerNorm_1/gamma/Initializer/ones:08
đ
"generator/G_MODEL/D/Conv/weights:0'generator/G_MODEL/D/Conv/weights/Assign'generator/G_MODEL/D/Conv/weights/read:02?generator/G_MODEL/D/Conv/weights/Initializer/truncated_normal:08
ķ
$generator/G_MODEL/D/LayerNorm/beta:0)generator/G_MODEL/D/LayerNorm/beta/Assign)generator/G_MODEL/D/LayerNorm/beta/read:026generator/G_MODEL/D/LayerNorm/beta/Initializer/zeros:08
đ
%generator/G_MODEL/D/LayerNorm/gamma:0*generator/G_MODEL/D/LayerNorm/gamma/Assign*generator/G_MODEL/D/LayerNorm/gamma/read:026generator/G_MODEL/D/LayerNorm/gamma/Initializer/ones:08
Á
$generator/G_MODEL/D/Conv_1/weights:0)generator/G_MODEL/D/Conv_1/weights/Assign)generator/G_MODEL/D/Conv_1/weights/read:02Agenerator/G_MODEL/D/Conv_1/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/D/LayerNorm_1/beta:0+generator/G_MODEL/D/LayerNorm_1/beta/Assign+generator/G_MODEL/D/LayerNorm_1/beta/read:028generator/G_MODEL/D/LayerNorm_1/beta/Initializer/zeros:08
Á
'generator/G_MODEL/D/LayerNorm_1/gamma:0,generator/G_MODEL/D/LayerNorm_1/gamma/Assign,generator/G_MODEL/D/LayerNorm_1/gamma/read:028generator/G_MODEL/D/LayerNorm_1/gamma/Initializer/ones:08
đ
"generator/G_MODEL/E/Conv/weights:0'generator/G_MODEL/E/Conv/weights/Assign'generator/G_MODEL/E/Conv/weights/read:02?generator/G_MODEL/E/Conv/weights/Initializer/truncated_normal:08
ķ
$generator/G_MODEL/E/LayerNorm/beta:0)generator/G_MODEL/E/LayerNorm/beta/Assign)generator/G_MODEL/E/LayerNorm/beta/read:026generator/G_MODEL/E/LayerNorm/beta/Initializer/zeros:08
đ
%generator/G_MODEL/E/LayerNorm/gamma:0*generator/G_MODEL/E/LayerNorm/gamma/Assign*generator/G_MODEL/E/LayerNorm/gamma/read:026generator/G_MODEL/E/LayerNorm/gamma/Initializer/ones:08
Á
$generator/G_MODEL/E/Conv_1/weights:0)generator/G_MODEL/E/Conv_1/weights/Assign)generator/G_MODEL/E/Conv_1/weights/read:02Agenerator/G_MODEL/E/Conv_1/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/E/LayerNorm_1/beta:0+generator/G_MODEL/E/LayerNorm_1/beta/Assign+generator/G_MODEL/E/LayerNorm_1/beta/read:028generator/G_MODEL/E/LayerNorm_1/beta/Initializer/zeros:08
Á
'generator/G_MODEL/E/LayerNorm_1/gamma:0,generator/G_MODEL/E/LayerNorm_1/gamma/Assign,generator/G_MODEL/E/LayerNorm_1/gamma/read:028generator/G_MODEL/E/LayerNorm_1/gamma/Initializer/ones:08
Á
$generator/G_MODEL/E/Conv_2/weights:0)generator/G_MODEL/E/Conv_2/weights/Assign)generator/G_MODEL/E/Conv_2/weights/read:02Agenerator/G_MODEL/E/Conv_2/weights/Initializer/truncated_normal:08
ū
&generator/G_MODEL/E/LayerNorm_2/beta:0+generator/G_MODEL/E/LayerNorm_2/beta/Assign+generator/G_MODEL/E/LayerNorm_2/beta/read:028generator/G_MODEL/E/LayerNorm_2/beta/Initializer/zeros:08
Á
'generator/G_MODEL/E/LayerNorm_2/gamma:0,generator/G_MODEL/E/LayerNorm_2/gamma/Assign,generator/G_MODEL/E/LayerNorm_2/gamma/read:028generator/G_MODEL/E/LayerNorm_2/gamma/Initializer/ones:08
Ų
*generator/G_MODEL/out_layer/Conv/weights:0/generator/G_MODEL/out_layer/Conv/weights/Assign/generator/G_MODEL/out_layer/Conv/weights/read:02Ggenerator/G_MODEL/out_layer/Conv/weights/Initializer/truncated_normal:08*ŧ
custom_signatureĶ
B
input9
generator_input:0"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸT
outputJ
"generator/G_MODEL/out_layer/Tanh:0"ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
AnimeGANv2