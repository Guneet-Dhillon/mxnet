package AI::MXNet::Optimizer;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::NDArray;
use AI::MXNet::Random;
use List::Util qw(max);

=head1  DESCRIPTION

Common Optimization algorithms with regularizations.

=cut

use Mouse;
use AI::MXNet::Function::Parameters;
my %opt_registry;
method get_opt_registry()
{
    return \%opt_registry;
}

method register()
{
    my $name = lc $self;
    ($name) = $name =~ /::(\w+)$/;
    if(exists $opt_registry{ $name })
    {
        my $existing = $opt_registry{ $name };
        warn(
            "WARNING: New optimizer $self.$name" 
            ."is overriding existing optimizer $existing.$name"
        );
    }
    $opt_registry{ $name } = $self;
}

=head2 create_optimizer

        Create an optimizer with specified name.

        Parameters
        ----------
        name: str
            Name of required optimizer. Should be the name
            of a subclass of Optimizer. Case insensitive.

        rescale_grad : float
            Rescaling factor on gradient. Normally should be 1/batch_size.

        kwargs: dict
            Parameters for optimizer

        Returns
        -------
        opt : Optimizer
            The result optimizer.
=cut

method create_optimizer(Str $name, %kwargs)
{
    if(exists $opt_registry{ lc $name })
    {
        my $rescale_grad = delete($kwargs{rescale_grad})//1;
        return $opt_registry{ lc $name }->new(
            rescale_grad => $rescale_grad,
            %kwargs
        );
    }
    confess("Cannot find optimizer $name");
}

*create = \&create_optimizer;

has 'rescale_grad'        => (is => "rw", isa => "Num", default=>1);
has 'lr'                  => (is => "rw", isa => "Num");
has 'learning_rate'       => (is => "rw", isa => "Num", default => 0.01);
has 'lr_scheduler'        => (is => "rw", isa => "Maybe[AI::MXNet::LRScheduler]");
has 'wd'                  => (is => "rw", isa => "Num", default => 0);
has 'lr_mult'             => (is => "rw", isa => "HashRef", default => sub { +{} });
has 'wd_mult'             => (is => "rw", isa => "HashRef", , default => sub { +{} });
has 'num_update'          => (is => "rw", isa => "Int");
has 'begin_num_update'    => (is => "rw", isa => "Int", default => 0);
has '_index_update_count' => (is => "rw", isa => "HashRef", default => sub { +{} });
has 'clip_gradient'       => (is => "rw", isa => "Maybe[Num]");
has 'param_idx2name'      => (is => "rw", isa => "HashRef[Str]", default => sub { +{} });
has 'idx2name'            => (is => "rw", isa => "HashRef[Str]");
has 'sym'                 => (is => "rw", isa => "Maybe[AI::MXNet::Symbol]");

sub BUILD
{
    my $self = shift;
    if($self->lr_scheduler)
    {
        $self->lr_scheduler->base_lr($self->learning_rate);
    }
    $self->lr($self->learning_rate);
    $self->num_update($self->begin_num_update);
    $self->idx2name({ %{ $self->param_idx2name } });
    $self->set_lr_mult({});
    $self->set_wd_mult({});
}
# Create additional optimizer state such as momentum.
# override in implementations.
method create_state($index, $weight){}

# Update the parameters. override in implementations
method update($index, $weight, $grad, $state){}

# set lr scale is deprecated. Use set_lr_mult instead.
method set_lr_scale($args_lrscale)
{
    Carp::cluck("set lr scale is deprecated. Use set_lr_mult instead.");
}

=head2 set_lr_mult

        Set individual learning rate multipler for parameters

        Parameters
        ----------
        args_lr_mult : dict of string/int to float
            set the lr multipler for name/index to float.
            setting multipler by index is supported for backward compatibility,
            but we recommend using name and symbol.
=cut

method set_lr_mult(HashRef[Num] $args_lr_mult)
{
    $self->lr_mult({});
    if($self->sym)
    {
        my $attr = $self->sym->attr_dict();
        for my $name (@{ $self->sym->list_arguments() })
        {
            if(exists $attr->{ $name } and exists $attr->{ $name }{ __lr_mult__ })
            {
                $self->lr_mult->{ $name } = $attr->{ $name }{ __lr_mult__ };
            }
        }
    }
    $self->lr_mult({ %{ $self->lr_mult }, %{ $args_lr_mult } });
}

=head2 set_wd_mult

        Set individual weight decay multipler for parameters.
        By default wd multipler is 0 for all params whose name doesn't
        end with _weight, if param_idx2name is provided.

        Parameters
        ----------
        args_wd_mult : dict of string/int to float
            set the wd multipler for name/index to float.
            setting multipler by index is supported for backward compatibility,
            but we recommend using name and symbol.
=cut

method set_wd_mult(HashRef[Num] $args_wd_mult)
{
    $self->wd_mult({});
    for my $n (values %{ $self->idx2name })
    {
        if(not $n =~ /(?:_weight|_gamma)$/)
        {
            $self->wd_mult->{ $n } = 0;
        }
    }
    if($self->sym)
    {
        my $attr = $self->sym->attr_dict();
        for my $name (@{ $self->sym->list_arguments() })
        {
            if(exists $attr->{ $name } and exists $attr->{ $name }{ __wd_mult__ })
            {
                $self->wd_mult->{ $name } = $attr->{ $name }{ __wd_mult__ };
            }
        }
    }
    $self->wd_mult({ %{ $self->wd_mult }, %{ $args_wd_mult } });
}

=head2 _update_count

        update num_update

        Parameters:
        index : int
            The index will be updated
=cut

method _update_count(Index $index)
{
    if(not exists $self->_index_update_count->{ $index })
    {
        $self->_index_update_count->{ $index } = $self->begin_num_update;
    }
    $self->_index_update_count->{ $index } += 1;
    $self->num_update(max($self->_index_update_count->{ $index }, $self->num_update));
}

=head2 _get_lr

        get learning rate for index.

        Parameters
        ----------
        index : int
            The index for weight

        Returns
        -------
        lr : float
            learning rate for this index
=cut

method _get_lr(Index $index)
{
    my $lr;
    if($self->lr_scheduler)
    {
        $lr = &{$self->lr_scheduler}($self->num_update);
    }
    else
    {
        $lr = $self->lr;
    }

    if(exists $self->lr_mult->{ $index })
    {
        $lr *= $self->lr_mult->{ $index };
    }
    elsif(exists $self->idx2name->{ $index })
    {
        $lr *= $self->lr_mult->{ $self->idx2name->{ $index } }//1;
    }
    return $lr;
}

=head2 _get_wd

        get weight decay for index.
        Returns 0 for non-weights if the name of weights are provided for __init__.

        Parameters
        ----------
        index : int
            The index for weight

        Returns
        -------
        wd : float
            weight decay for this index
=cut

method _get_wd(Index $index)
{
    my $wd = $self->wd;
    if(exists $self->wd_mult->{ $index })
    {
        $wd *= $self->wd_mult->{ $index };
    }
    elsif(exists $self->idx2name->{ $index })
    {
        $wd *= $self->wd_mult->{ $self->idx2name->{ $index } }//1;
    }
    return $wd;
}

=begin

    A very simple SGD optimizer with momentum and weight regularization.

    Parameters
    ----------
    learning_rate : float, optional
        learning_rate of SGD

    momentum : float, optional
       momentum value

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]

    param_idx2name : dict of string/int to float, optional
        special treat weight decay in parameter ends with bias, gamma, and beta
=cut

package AI::MXNet::SGD;
use Mouse;
extends 'AI::MXNet::Optimizer';

has 'kwargs'   => (is => "rw", isa => "HashRef[Num]");
has 'momentum' => (is => "rw", isa => "Num", default => 0);

sub BUILD
{
    my $self = shift;
    $self->kwargs({ rescale_grad => $self->rescale_grad });
    if($self->momentum)
    {
        $self->kwargs->{momentum} = $self->momentum;
    }
    if($self->clip_gradient)
    {
        $self->kwargs->{clip_gradient} = $self->clip_gradient;
    }
}

=head2 create_state

    Create additional optimizer state such as momentum.

        Parameters
        ----------
        weight : NDArray
            The weight data
=cut

method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    if($self->momentum == 0)
    {
        return undef;
    }
    else
    {
        return AI::MXNet::NDArray->zeros(
            $weight->shape, ctx => $weight->context, dtype => $weight->dtype
        );
    }
}

=head2 update

        Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
=cut

method update(
    Index                     $index,
    AI::MXNet::NDArray        $weight,
    AI::MXNet::NDArray        $grad,
    Maybe[AI::MXNet::NDArray] $state
)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    if($state)
    {
        AI::MXNet::NDArray->sgd_mom_update(
            $weight, $grad, $state,
            {
                out => $weight,
                lr  => $lr,
                wd  => $wd,
                %{ $self->kwargs }
            }
        );
    }
    else
    {
        AI::MXNet::NDArray->sgd_update(
            $weight,
            $grad,
            {
                out => $weight,
                lr  => $lr,
                wd  => $wd,
                %{ $self->kwargs }
            }
        );
    }
}

__PACKAGE__->register;

package AI::MXNet::DCASGD;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::Optimizer';

=head1 NAME

    AI::MXNet::DCASGD
=cut

=head1 DESCRIPTION

    DCASGD optimizer with momentum and weight regularization.

    implement paper "Asynchronous Stochastic Gradient Descent with
                    Delay Compensation for Distributed Deep Learning"

    Parameters
    ----------
    learning_rate : float, optional
        learning_rate of SGD

    momentum : float, optional
       momentum value

    lamda : float, optional
       scale DC value

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]

    param_idx2name : hash ref of string/int to float, optional
        special treat weight decay in parameter ends with bias, gamma, and beta
=cut
has 'momentum'        => (is => 'ro', isa => 'Num', default => 0);
has 'lamda'           => (is => 'ro', isa => 'Num', default => 0.04);
has 'weight_previous' => (is => 'rw', init_arg => undef);

sub BUILD
{
    my $self = shift;
    $self->weight_previous({});
}

=head2 create_state

    Create additional optimizer state such as momentum.

        Parameters
        ----------
        weight : NDArray
            The weight data
=cut

method create_state(Index $index, AI::MXNet::NDArray $weight)
{
        return [
            $self->momentum ? AI::MXNet::NDArray->zeros(
                $weight->shape, ctx => $weight->context, dtype => $weight->dtype
            ) : undef,
            $weight->copy
        ];
}

=head2 update

        Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
=cut

method update(
    Index                     $index,
    AI::MXNet::NDArray        $weight,
    AI::MXNet::NDArray        $grad,
    Maybe[AI::MXNet::NDArray] $state
)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    $grad *= $self->rescale_grad;
    if($self->clip_gradient)
    {
        $grad = AI::MXNet::NDArray->clip(
            $grad,
            -$self->clip_gradient,
            $self->clip_gradient
        );
    }
    my ($mom, $weight_previous) = @{ $state };
    if(defined $mom)
    {
        $mom *= $self->momentum;
        $mom += -$lr * (
                $grad + $wd * $weight
                    +
                $self->lamda * $grad * $grad * ($weight - $weight_previous)
        );
    }
    else
    {
        assert($self->momentum == 0);
        $mom = -$lr * (
                $grad + $wd * $weight
                    +
                $self->lamda * $grad * $grad * ($weight - $weight_previous)
        );
    }
    $weight_previous .= $weight;
    $weight += $mom;
}

__PACKAGE__->register;

=begin
    SGD with nesterov
    It is implemented according to
    https://github.com/torch/optim/blob/master/sgd.lua
=cut

package AI::MXNet::NAG;
use Mouse;

extends 'AI::MXNet::SGD';

=head2 update

        Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
=cut

method update(
    Index $index,
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    AI::MXNet::NDArray|Undef $state
)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    $grad = $grad * $self->rescale_grad;
    if($self->clip_gradient)
    {
        $grad = AI::MXNet::NDArray->clip(
            $grad, 
            -$self->clip_gradient,
            $self->clip_gradient
        );
    }
    if($state)
    {
        my $mom  = $state;
        $mom    *= $self->momentum;
        $grad   += $wd * $weight;
        $mom    += $grad;
        $grad   += $self->momentum * $mom;
        $weight += -$lr * $grad;
    }
    else
    {
        confess("momentum != 0") unless $self->momentum == 0;
        $weight += -$lr * ($grad + $wd * $weight);
    }
}

__PACKAGE__->register;

=begin
    Stochastic Langevin Dynamics Updater to sample from a distribution.

    Parameters
    ----------
    learning_rate : float, optional
        learning_rate of SGD

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]

    param_idx2name : dict of string/int to float, optional
        special treat weight decay in parameter ends with bias, gamma, and beta
=cut

package AI::MXNet::SLGD;
use Mouse;

extends 'AI::MXNet::Optimizer';

=head2 create_state

        Create additional optimizer state such as momentum.

        Parameters
        ----------
        weight : NDArray
            The weight data
=cut

method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return undef;
}

=head2 update

        Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
=cut

method update(
    Index $index, 
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    AI::MXNet::NDArray|Undef $state
)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    $grad *= $self->rescale_grad;
    if($self->clip_gradient)
    {
        $grad = AI::MXNet::NDArray->clip(
            $grad,
            -$self->clip_gradient,
             $self->clip_gradient
        );
    }
    $weight +=  - $lr/2 * ($grad + $wd * $weight)
                    +
                AI::MXNet::Random->normal(
                        0, sqrt($lr),
                        $weight->shape,
                        $weight->context
                );
}

__PACKAGE__->register;

=begin

    Adam optimizer as described in [King2014]_.

    .. [King2014] Diederik Kingma, Jimmy Ba,
       *Adam: A Method for Stochastic Optimization*,
       http://arxiv.org/abs/1412.6980

    the code in this class was adapted from
    https://github.com/mila-udem/blocks/blob/master/blocks/algorithms/__init__.py#L765

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.001.
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
        Default value is set to 0.9.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
        Default value is set to 0.999.
    epsilon : float, optional
        Default value is set to 1e-8.
    decay_factor : float, optional
        Default value is set to 1 - 1e-8.

    wd : float, optional
        L2 regularization coefficient add to all the weights
    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
=cut
package AI::MXNet::Adam;
use Mouse;

extends 'AI::MXNet::Optimizer';

has 'kwargs'   => (is => "rw", isa => "HashRef[Num]");
has '+learning_rate' => (default => 0.001);
has 'beta1'    => (is => "rw", isa => "Num", default => 0.9);
has 'beta2'    => (is => "rw", isa => "Num", default => 0.999);
has 'epsilon'  => (is => "rw", isa => "Num", default => 1e-8);
has 'decay_factor'  => (is => "rw", isa => "Num", default => (1 - 1e-8));

sub BUILD
{
    my $self = shift;
    $self->kwargs({
        rescale_grad => $self->rescale_grad,
        beta1   => $self->beta1,
        beta2   => $self->beta2,
        epsilon => $self->epsilon
    });
    if($self->clip_gradient)
    {
        $self->kwargs->{clip_gradient} = $self->clip_gradient;
    }
}
=head2 create_state

        Create additional optimizer state: mean, variance

        Parameters
        ----------
        weight : NDArray
            The weight data
=cut

method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return [AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context,
                dtype => $weight->dtype
            ),  # mean
            AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context,
                dtype => $weight->dtype
            )  # variance
    ];
}

=head2 update

        Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
=cut

method update(
    Index $index, 
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    ArrayRef[AI::MXNet::NDArray] $state
)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    my $t = $self->_index_update_count->{$index};
    my $coef1 = 1 - $self->beta1**$t;
    my $coef2 = 1 - $self->beta2**$t;
    $lr *= sqrt($coef2)/$coef1;
    my ($mean, $var) = @{ $state };
    AI::MXNet::NDArray->adam_update(
        $weight, $grad, $mean, $var,
        {
            out => $weight,
            lr  => $lr,
            wd  => $wd,
            %{ $self->kwargs }
        }
    );
}

__PACKAGE__->register;

=begin

    AdaGrad optimizer of Duchi et al., 2011,

    This code follows the version in http://arxiv.org/pdf/1212.5701v1.pdf  Eq(5)
    by Matthew D. Zeiler, 2012. AdaGrad will help the network to converge faster
    in some cases.

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.05.

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.

    eps: float, optional
        A small float number to make the updating processing stable
        Default value is set to 1e-7.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
=cut
package AI::MXNet::AdaGrad;
use Mouse;

extends 'AI::MXNet::Optimizer';

has 'float_stable_eps'    => (is => "rw", isa => "Num", default => 1e-7);
has '+learning_rate'       => (default => 0.05);

method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return AI::MXNet::NDArray->zeros(
                $weight->shape, 
                ctx => $weight->context
    );  # history
}

method update(
    Index $index,
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    AI::MXNet::NDArray $state
)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    $grad *= $self->rescale_grad;
    if($self->clip_gradient)
    {
        $grad = AI::MXNet::NDArray->clip(
            $grad,
            -$self->clip_gradient,
             $self->clip_gradient
        );
    }
    my $history = $state;
    $history += ($grad * $grad);
    $weight  += -$lr
                    *
                (
                    $grad
                        /
                    AI::MXNet::NDArray->sqrt(
                        $history
                            +
                        $self->float_stable_eps
                    )
                        +
                    $wd * $weight
                );
}

__PACKAGE__->register;

=begin

    RMSProp optimizer of Tieleman & Hinton, 2012,

    For centered=False, the code follows the version in
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf by
    Tieleman & Hinton, 2012

    For centered=True, the code follows the version in
    http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.001.
    gamma1: float, optional
        decay factor of moving average for gradient^2.
        Default value is set to 0.9.
    gamma2: float, optional
        "momentum" factor.
        Default value if set to 0.9.
        Only used if centered=True
    epsilon : float, optional
        Default value is set to 1e-8.
    centered : bool, optional
        Use Graves or Tielemans & Hintons version of RMSProp
    wd : float, optional
        L2 regularization coefficient add to all the weights
    rescale_grad : float, optional
        rescaling factor of gradient.
    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
    clip_weights : float, optional
        clip weights in range [-clip_weights, clip_weights]
=cut

package AI::MXNet::RMSProp;
use Mouse;

extends 'AI::MXNet::Optimizer';

has '+learning_rate' => (default => 0.001);
has 'gamma1'         => (is => "ro", isa => "Num",  default => 0.9);
has 'gamma2'         => (is => "ro", isa => "Num",  default => 0.9);
has 'epsilon'        => (is => "ro", isa => "Num",  default => 1e-8);
has 'centered'       => (is => "ro", isa => "Bool", default => 0);
has 'clip_weights'   => (is => "ro", isa => "Num");
has 'kwargs'         => (is => "rw", init_arg => undef);

sub BUILD
{
    my $self = shift;
    $self->kwargs({
        rescale_grad => $self->rescale_grad,
        gamma1       => $self->gamma1,
        epsilon      => $self->epsilon
    });
    if($self->centered)
    {
        $self->kwargs->{gamma2} = $self->gamma2;
    }
    if($self->clip_gradient)
    {
        $self->kwargs->{clip_gradient} = $self->clip_gradient;
    }
    if($self->clip_weights)
    {
        $self->kwargs->{clip_weights} = $self->clip_weights;
    }
}

# For centered=False: n
# For centered=True: n, g, delta
method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return [
            $self->centered
            ? (
                AI::MXNet::NDArray->zeros(
                    $weight->shape,
                    ctx => $weight->context
                ),  # n
                AI::MXNet::NDArray->zeros(
                    $weight->shape,
                    ctx => $weight->context
                ),  # g
                AI::MXNet::NDArray->zeros(
                    $weight->shape,
                    ctx => $weight->context
                )
            )   # delta
            : (
                AI::MXNet::NDArray->zeros(
                    $weight->shape,
                    ctx => $weight->context
                ),  # n
            )
    ];
}

method update(
    Index $index,
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    ArrayRef[AI::MXNet::NDArray] $state
)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    my ($n, $g, $delta) = @{ $state };
    if($self->centered)
    {
        AI::MXNet::NDArray->rmspropalex_update(
            $weight, $grad, $n, $g, $delta,
            {
                out => $weight,
                lr  => $lr,
                wd  => $wd,
                %{ $self->kwargs }
            }
        );
    }
    else
    {
        AI::MXNet::NDArray->rmsprop_update(
            $weight, $grad, $n,
            {
                out => $weight,
                lr  => $lr,
                wd  => $wd,
                %{ $self->kwargs }
            }
        );
    }
}

__PACKAGE__->register;

=begin

    AdaDelta optimizer as described in
    Zeiler, M. D. (2012).
    *ADADELTA: An adaptive learning rate method.*

    http://arxiv.org/abs/1212.5701

    Parameters
    ----------
    rho: float
        Decay rate for both squared gradients and delta x
    epsilon : float
        The constant as described in the thesis
    wd : float
        L2 regularization coefficient add to all the weights
    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.
    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
=cut
package AI::MXNet::AdaDelta;
use Mouse;

extends 'AI::MXNet::Optimizer';

has 'rho'    => (is => "rw", isa => "Num", default => 0.9);
has 'epsilon'    => (is => "rw", isa => "Num", default => 1e-5);

method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return [
            AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context
            ),  # accumulated g
            AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context
            )   # accumulated delta
    ];
}

method update(
    Index $index,
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    ArrayRef[AI::MXNet::NDArray] $state
)
{
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    $grad *= $self->rescale_grad;
    if($self->clip_gradient)
    {
        $grad = AI::MXNet::NDArray->clip(
            $grad,
            -$self->clip_gradient,
             $self->clip_gradient
        );
    }
    my ($acc_g, $acc_delta) = @{ $state };
    $acc_g .= $self->rho * $acc_g + (1 - $self->rho) * $grad * $grad;
    my $current_delta = ($acc_delta + $self->epsilon)->sqrt
                            /
                        ($acc_g + $self->epsilon)->sqrt
                            *
                        $grad;
    $acc_delta .= $self->rho * $acc_delta + (1 - $self->rho) * $current_delta * $current_delta;
    $weight -= $current_delta + $wd * $weight;
}

__PACKAGE__->register;

# For test use
package AI::MXNet::Test;
use Mouse;

extends 'AI::MXNet::Optimizer';

# Create a state to duplicate weight
method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return AI::MXNet::NDArray->zeros(
                $weight->shape, 
                ctx => $weight->context
    );
}

# performs w += rescale_grad * grad
method update(
    Index $index,
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    AI::MXNet::NDArray $state
)
{
    $weight += $grad * $self->rescale_grad;
    $state .= $weight;
}

__PACKAGE__->register;

# updater for kvstore
package AI::MXNet::Updater;
use Mouse;
use Storable qw(thaw freeze);
use overload "&{}" => sub { my $self = shift; sub { $self->call(@_) } },
             fallback => 1;

has "optimizer" => (is => "rw", isa => "AI::MXNet::Optimizer");
has "states"    => (is => "rw", isa => "HashRef", default => sub { +{} });

method call(Index $index, AI::MXNet::NDArray $grad, AI::MXNet::NDArray $weight)
{
    if(not exists $self->states->{ $index })
    {
        $self->states->{ $index } = $self->optimizer->create_state($index, $weight);
    }
    $self->optimizer->update($index, $weight, $grad, $self->states->{ $index });
}
*slice = *call;

method set_states($states)
{
    $self->states(thaw($states));
}

method get_states()
{
    return freeze($self->states);
}

=begin

Return a closure of the updater needed for kvstore

    Parameters
    ----------
    optimizer: Optimizer
         The optimizer

    Returns
    -------
    updater: function
         The closure of the updater
=cut

package AI::MXNet::Optimizer;


method get_updater(AI::MXNet::Optimizer $optimizer)
{
    return AI::MXNet::Updater->new(optimizer => $optimizer);
}

1;