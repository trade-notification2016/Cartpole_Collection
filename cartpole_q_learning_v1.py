import numpy as np
import sys
import random
import tensorflow as tf
import itertools
slim = tf.contrib.slim

BATCH_SIZE = 32
TRAIN_ITER = 1000
EARLY_TRAIN_TERMINATE_CRITERIA = 1e-3;
LEARNING_RATE = 0.001
BETA = 0.9

class Estimator():
    def __init__(self,env):
        state = tf.placeholder(tf.float32,[None]+list(env.observation_space.sample().shape))
        q_target_action = tf.placeholder(tf.int32,[None,])
        q_target_val = tf.placeholder(tf.float32,[None,])

        net = slim.fully_connected(state, 50, scope='fc1')
        net = slim.fully_connected(net, 50, scope='fc2')
        net = slim.fully_connected(net, env.action_space.n, activation_fn=None, scope='q_s_a')

        cat_idx = tf.stack([tf.range(0, tf.shape(net)[0]), q_target_action], axis=1)
        error = q_target_val - tf.gather_nd(net,cat_idx)

        #loss = tf.reduce_mean(error**2)
        #Instead of raw loss value, let's try to use Huber loss.
        #Then, gradient will be naturally clipped to -1 to 1.
        loss = tf.reduce_mean(tf.where(tf.abs(error) < 1.0,
                                       0.5 * tf.square(error),
                                       tf.abs(error) - 0.5))

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE,beta1=BETA) #works best;
        #optimizer = tf.train.MomentumOptimizer(LEARNING_RATE,BETA)
        #optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(loss)

        self.state = state
        self.q_target_action = q_target_action
        self.q_target_val = q_target_val
        self.q_s_a = net
        self.loss = loss
        self.train = train_op

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        return state

    def initialize(self):
        if hasattr(self,'sess'): return

        sess = tf.Session()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        self.sess=sess

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        if( len(s.shape) == 1 ) :
            s = np.expand_dims(s,axis=0)
        q_s_a = self.sess.run(self.q_s_a,
                              feed_dict={self.state: self.featurize_state(s)})

        return q_s_a[:,a] if a is not None else q_s_a

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        if( len(s.shape) == 1 ) :
            s = np.expand_dims(s,axis=0)
            a = np.expand_dims(a,axis=0)
            y = np.expand_dims(y,axis=0)

        assert(a.shape == y.shape)
        #print a[:3],self.predict(s)[range(0,3),a[:3]],y[:3]

        loss,_ = self.sess.run([self.loss,self.train],
                              feed_dict={self.state: self.featurize_state(s),
                                         self.q_target_action: a,
                                         self.q_target_val: y})
        return loss

def make_epsilon_greedy_policy(Q_estimator, epsilon, nA):
    def policy_fn(observation):
        assert( len(observation.shape) == 1 )
        if( callable(epsilon) ) :
            eps = epsilon()
        else :
            eps = epsilon

        if( random.random() < eps ):
            return np.ones([nA,]) / nA
        else :
            q_s_a = Q_estimator.predict(observation)[0]
            return np.eye(nA)[np.argmax(q_s_a)]
    return policy_fn

class ReplayMemory() :
    def __init__(self,max_len=5000):
        self.max_len = max_len
        self.buf = []

    def add_memory(self,s,a,r,s_):
        if(len(self.buf) >= self.max_len) :
            self.buf[random.randint(0,self.max_len-1)] = (s,a,r,s_,None)
        else :
            self.buf.append((s,a,r,s_,None))

    def update_q_target(self,q,discount_factor):
        for i in range(0,self.length(),512) :
            _, _, rewards, next_states, _ = [np.array(x) for x in map(list, zip(*self.buf[i:min(i+512,self.length())]))]
            q_targets = rewards + discount_factor * np.amax(q.predict(next_states),axis=1)
            for j,val in enumerate(q_targets) :
                #val = max(val ,self.buf[i+j][4])
                self.buf[i+j] = self.buf[i+j][:4]+(val,)

    def sample(self,sample_num) :
        return random.sample(self.buf,sample_num)

    def length(self) :
        return len(self.buf)

def q_learning(env, Q_estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0, min_epsilon=0.01):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The policy we're following
    greedy_policy = make_epsilon_greedy_policy(Q_estimator,0.,env.action_space.n)
    epsilon_greedy_policy = make_epsilon_greedy_policy(Q_estimator,
                                                       lambda : max(min_epsilon,epsilon * epsilon_decay**i_episode),
                                                       env.action_space.n)
    def pick_action(policy,s) :
        pi_given_s = policy(s)
        c = np.random.choice(env.action_space.n, 1, p=pi_given_s)[0]
        return c

    memory = ReplayMemory()
    results = []
    for i_episode in range(num_episodes):
        state = env.reset()

        loss = 0.
        for i_frame in itertools.count() :
            action = pick_action(epsilon_greedy_policy,state)
            next_state, reward, done, _ = env.step(action)
            # reward reshaping is the key!
            reward = 0.
            if( done ) :
                if( i_frame > 400 ):
                    reward = 1.
                elif( i_frame > 200 ) :
                    reward = 0.
                else :
                    reward = -1.
                #reward = min(1.,(i_frame - 100) / 100.)

            memory.add_memory(state,action,reward,next_state)

            if(done):
                results.append(i_frame+1)
                print("Episode %4d/%4d %3d"%(i_episode + 1, num_episodes, i_frame+1))
                sys.stdout.flush()
                if( 1. * sum(results[-100:])/len(results[-100:]) > 475. ) :
                    print("Solved %d"%(i_episode-99))
                    return
                break
            state = next_state

        if( memory.length() >= BATCH_SIZE ) :
            memory.update_q_target(Q_estimator,discount_factor) #fix q;
            for _ in range(TRAIN_ITER):
                states, actions, rewards, next_states, q_targets = [np.array(x) for x in map(list, zip(*memory.sample(BATCH_SIZE)))]
                # use pre-calculated(fixed) q value.
                #q_targets = rewards + discount_factor * np.amax(Q_estimator.predict(next_states),axis=1)
                loss = Q_estimator.update(states,actions,q_targets)
                if( loss < EARLY_TRAIN_TERMINATE_CRITERIA ) : break

        # Evaluation with Greedy Policy.
        #rewards = 0
        #for _ in range(10):
        #    state = env.reset()
        #    reward = 0.; done = False
        #    while(not done):
        #        action = pick_action(greedy_policy,state)
        #        state, r, done, _ = env.step(action)
        #        reward = r + discount_factor * reward
        #    rewards += reward/10.
        # Print out which episode we're on, useful for debugging.
        #print("Episode %d/%d %f; %f"%(i_episode + 1, num_episodes, rewards, loss))
        #sys.stdout.flush()
    return

if __name__ == '__main__' :
    import gym
    from gym import wrappers
    env = gym.make('CartPole-v1')
    env = wrappers.Monitor(env, '/tmp/cartpole',force=True)

    estimator = Estimator(env)
    estimator.initialize()
    q_learning(env, estimator, 500, discount_factor=1.0, epsilon=1.0, epsilon_decay=0.9, min_epsilon=0.)