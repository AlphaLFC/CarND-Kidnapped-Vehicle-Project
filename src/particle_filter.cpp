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

ParticleFilter::ParticleFilter(int num_particles)
{
	this->num_particles = num_particles;
	this->gaussian_x = NULL;
	this->gaussian_y = NULL;
	this->gaussian_theta = NULL;
}

ParticleFilter::~ParticleFilter()
{
	this->free_gaussians();
}

void ParticleFilter::free_gaussians()
{
	if (this->gaussian_x != NULL)
	{
		delete this->gaussian_x;
	}
	if (this->gaussian_y != NULL)
	{
		delete this->gaussian_y;
	}
	if (this->gaussian_theta != NULL)
	{
		delete this->gaussian_theta;
	}
}

void ParticleFilter::update_pos_std(double pos_std[])
{
	this->free_gaussians();
	this->gaussian_x = new normal_distribution<double>(0.0, pos_std[0]);	  // GPS measurement uncertainty x [m]
	this->gaussian_y = new normal_distribution<double>(0.0, pos_std[1]);	  // GPS measurement uncertainty y [m]
	this->gaussian_theta = new normal_distribution<double>(0.0, pos_std[2]); // GPS measurement uncertainty theta [rad]
}

void ParticleFilter::init(double x, double y, double theta, double pos_std[])
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	this->update_pos_std(pos_std);

	for (int i = 0; i < this->num_particles; i++)
	{
		double noise_x = (*this->gaussian_x)(gen);
		double noise_y = (*this->gaussian_y)(gen);
		double noise_theta = (*this->gaussian_theta)(gen);
		this->weights.push_back(1.);
		this->particles.push_back(Particle(i, x + noise_x, y + noise_y, theta + noise_theta, 1.));
	}
}

void ParticleFilter::prediction(double delta_t, double pos_std[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	this->update_pos_std(pos_std);
	
	for (int i = 0; i < this->num_particles; i++)
	{
		this->particles[i].move(delta_t, velocity, yaw_rate);
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
}

void ParticleFilter::resample()
{
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
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
