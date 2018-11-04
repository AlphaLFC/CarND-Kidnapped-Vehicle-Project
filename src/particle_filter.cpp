/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;


// void ParticleFilter::free_gaussians()
// {
// 	if (this->gaussian_x != NULL)
// 	{
// 		delete this->gaussian_x;
// 	}
// 	if (this->gaussian_y != NULL)
// 	{
// 		delete this->gaussian_y;
// 	}
// 	if (this->gaussian_theta != NULL)
// 	{
// 		delete this->gaussian_theta;
// 	}
// }

// void ParticleFilter::update_pos_std(double pos_std[])
// {
// 	this->free_gaussians();
// 	this->gaussian_x = new normal_distribution<double>(0.0, pos_std[0]);	  // GPS measurement uncertainty x [m]
// 	this->gaussian_y = new normal_distribution<double>(0.0, pos_std[1]);	  // GPS measurement uncertainty y [m]
// 	this->gaussian_theta = new normal_distribution<double>(0.0, pos_std[2]); // GPS measurement uncertainty theta [rad]
// }

void ParticleFilter::init(double x, double y, double theta, double pos_std[])
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	// this->update_pos_std(pos_std);

	normal_distribution<double> gaussian_x(0.0, pos_std[0]);	 // GPS measurement uncertainty x [m]
	normal_distribution<double> gaussian_y(0.0, pos_std[1]);	 // GPS measurement uncertainty y [m]
	normal_distribution<double> gaussian_theta(0.0, pos_std[2]); // GPS measurement uncertainty theta [rad]
	default_random_engine gen;

	for (int i = 0; i < this->num_particles; i++)
	{
		double noise_x = gaussian_x(gen);
		double noise_y = gaussian_y(gen);
		double noise_theta = gaussian_theta(gen);
		/* this->weights.push_back(1.); */
		this->particles.push_back(Particle(i, x + noise_x, y + noise_y, theta + noise_theta, 1.));
	}
}

void ParticleFilter::prediction(double delta_t, double pos_std[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> gaussian_x(0.0, pos_std[0]);	 // GPS measurement uncertainty x [m]
	normal_distribution<double> gaussian_y(0.0, pos_std[1]);	 // GPS measurement uncertainty y [m]
	normal_distribution<double> gaussian_theta(0.0, pos_std[2]); // GPS measurement uncertainty theta [rad]
	default_random_engine gen;

	for (int i = 0; i < this->num_particles; i++)
	{
		this->particles[i].move(delta_t, velocity, yaw_rate, gaussian_x, gaussian_y, gaussian_theta, gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	std::vector<double> probs = std::vector<double>();
    const std::vector<LandmarkObs>& obsvs = observations;
    const std::vector<Map::single_landmark_s>& lms = map_landmarks.landmark_list; 
    float sum_weights = 0.0;

	for (size_t i = 0; i < this->particles.size(); i++)
	{
		double prob = 1.0;
		std::vector<bool> associated = std::vector<bool>(false, lms.size());
		for (size_t j = 0; j < obsvs.size(); j++)
		{
			double obs_range = sqrt(obsvs[j].x * obsvs[j].x + obsvs[j].y * obsvs[j].y);
			if ( obs_range > sensor_range ) 
			{
				continue;
			}
			LandmarkObs obs_g_pos = affine_transform(obsvs[j], this->particles[i].theta, 
													 this->particles[i].x, this->particles[i].y);
			
			double min_dist = 999999999;
			double min_lm_idx = -1;
			for(size_t m = 0; m < lms.size(); m++)
			{
                if (associated[m]) 
                {
                    continue;
                }

				double d = dist(obs_g_pos.x, obs_g_pos.y, lms[m].x_f, 
				                lms[m].y_f);
				if (d < min_dist) {
					min_dist = d;
					min_lm_idx = m;
				}
			}
			associated[min_lm_idx] = true;

			// TODO: use normpdf to calculate the prob of observation and the associated landmark.
            prob *= normpdf2d(obs_g_pos.x, obs_g_pos.y, lms[min_lm_idx].x_f, lms[min_lm_idx].y_f, std_landmark[0], std_landmark[1]);
		}
        this->particles[i].weight = prob;
        sum_weights += prob;
	}

    for (size_t i = 0; i < this->particles.size(); i++) {
        this->particles[i].weight /= sum_weights;
    }
}

void ParticleFilter::resample()
{
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::default_random_engine gen;
    std::vector<float> weights;
    std::vector<Particle> new_particles;
    std::discrete_distribution<> d(weights.begin(), weights.end());
    
    for (size_t i = 0; i < this->particles.size(); i++) {
        weights.push_back(this->particles[i].weight);
    }
    
    for (size_t i = 0; i < this->particles.size(); i++) {
        int random_gen_idx = d(gen);
        new_particles.push_back(this->particles[random_gen_idx]);
    }

    this->particles.clear();
    this->particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
										 const std::vector<double> &sense_x, const std::vector<double> &sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
