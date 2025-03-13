package utils

import (
	"context"
	"encoding/gob"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"sync"
	"time"

	"github.com/golang/geo/r3"
	"go.viam.com/rdk/components/arm"
	"go.viam.com/rdk/components/posetracker"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/motionplan/ik"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/spatialmath"
	"go.viam.com/utils"

	commonpb "go.viam.com/api/common/v1"
)

const (
	randSeed               = 0
	orientationScaleFactor = 10.
	iterations             = 10000
)

var limits = []referenceframe.Limit{
	// R3 Vector
	{Min: -500, Max: 500}, // X(mm)
	{Min: -500, Max: 500}, // Y(mm)
	{Min: -500, Max: 500}, // Z(mm)
	// R3AA
	{Min: -2 * math.Pi, Max: 2 * math.Pi}, // X
	{Min: -2 * math.Pi, Max: 2 * math.Pi}, // Y
	{Min: -2 * math.Pi, Max: 2 * math.Pi}, // Z
}

var logger = logging.NewLogger("client")

func DiscoverTags(ctx context.Context, poseTracker posetracker.PoseTracker) ([]string, error) {
	poses, err := poseTracker.Poses(ctx, nil, nil)
	if err != nil {
		return nil, err
	}
	tags := []string{}
	for key, _ := range poses {
		tags = append(tags, key)
	}
	return tags, nil
}

func EstimateFramePose(
	ctx context.Context,
	a arm.Arm,
	pt posetracker.PoseTracker,
	expectedTags []string,
	calibrationPositions [][]referenceframe.Input,
	seedPose spatialmath.Pose,
	collectNewData bool,
) (spatialmath.Pose, error) {
	var tagPoses []referenceframe.FrameSystemPoses
	var err error
	if collectNewData {
		tagPoses, _, err = getTagPoses(ctx, a, pt, calibrationPositions)
		if err != nil {
			return nil, err
		}
		if err = saveToFile("poses.gob", posesToProtobuf(tagPoses)); err != nil {
			return nil, err
		}
		if err = saveToFile("configurations.gob", calibrationPositions); err != nil {
			return nil, err
		}
	}

	var tagPosesFromFile []map[string]*commonpb.Pose
	if err := loadFromFile("poses.gob", &tagPosesFromFile); err != nil {
		return nil, err
	}
	var calibrationPositionsFromFile [][]referenceframe.Input
	if err := loadFromFile("configurations.gob", &calibrationPositionsFromFile); err != nil {
		return nil, err
	}
	printWorldStatePoses(a, protobufToPoses(tagPosesFromFile), seedPose, expectedTags, calibrationPositionsFromFile)

	sol, err := minimize(ctx, a.ModelFrame(), protobufToPoses(tagPosesFromFile), expectedTags, calibrationPositionsFromFile, seedPose)
	if err != nil {
		return nil, err
	}
	fmt.Println("Guess:", seedPose.Point(), seedPose.Orientation().Quaternion())
	p := floatsToPose(sol[0].q)
	fmt.Println(p.Point(), p.Orientation().Quaternion(), sol[0].cost)
	printWorldStatePoses(a, protobufToPoses(tagPosesFromFile), p, expectedTags, calibrationPositionsFromFile)
	return p, nil
}

type basicNode struct {
	q    []float64
	cost float64
}

func minimize(
	ctx context.Context,
	model referenceframe.Model,
	poses []referenceframe.FrameSystemPoses,
	tags []string,
	calibrationPositions [][]referenceframe.Input,
	seedPose spatialmath.Pose,
) ([]basicNode, error) {
	lossFunction := func(input []float64) float64 {
		testPose := floatsToPose(input)

		// make a map of tags to the pose they are measured at in world coordinates
		tagPoseMap := make(map[string][]spatialmath.Pose)
		for i, pose := range poses {
			armPose, err := model.Transform(calibrationPositions[i])
			if err != nil {
				panic(err)
			}
			for _, tag := range tags {
				p, ok := pose[tag]
				if !ok {
					continue
				}
				tagPoseMap[tag] = append(tagPoseMap[tag], spatialmath.Compose(spatialmath.Compose(armPose, testPose), p.Pose()))
			}
		}

		// sum distances over all correspondences
		cumSum := 0.
		for _, tag := range tags {
			tagPoses := tagPoseMap[tag]
			n := len(tagPoses)
			count := (n - 1) * n / 2 // formula to calculate the ith triangle number
			for i := range n - 1 {
				for j := i + 1; j < n; j++ {
					delta := spatialmath.PoseDelta(tagPoses[i], tagPoses[j])
					tagDist := delta.Point().Norm2()
					tagDist += spatialmath.QuatToR3AA(delta.Orientation().Quaternion()).Mul(orientationScaleFactor).Norm2()
					cumSum += tagDist / float64(count)
				}
			}
		}
		// fmt.Println(cumSum, input)
		return cumSum / float64(len(tags))
	}

	solver, err := ik.CreateNloptSolver(limits, logger, iterations, false, false)
	if err != nil {
		return nil, err
	}

	solutionGen := make(chan *ik.Solution, 1)
	ikErr := make(chan error, 1)

	var activeSolvers sync.WaitGroup
	defer activeSolvers.Wait()
	activeSolvers.Add(1)

	ctxWithCancel, cancel := context.WithCancel(ctx)
	defer cancel()

	// Spawn the IK solver to generate solutions until done
	utils.PanicCapturingGo(func() {
		defer close(ikErr)
		defer activeSolvers.Done()
		ikErr <- solver.Solve(ctxWithCancel, solutionGen, poseToFloats(seedPose), lossFunction, randSeed)
	})

	solutions := map[float64][]float64{}

	// Solve the IK solver. Loop labels are required because `break` etc in a `select` will break only the `select`.
IK:
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		select {
		case solution := <-solutionGen:
			solutions[solution.Score] = solution.Configuration
			continue IK
		default:
		}

		select {
		case <-ikErr:
			// If we have a return from the IK solver, there are no more solutions, so we finish processing above
			// until we've drained the channel, handled by the `continue` above
			break IK
		default:
		}
	}

	// Cancel any ongoing processing within the IK solvers if we're done receiving solutions
	cancel()
	for done := false; !done; {
		select {
		case <-solutionGen:
		default:
			done = true
		}
	}

	if len(solutions) == 0 {
		return nil, errors.New("no solutions found")
	}

	keys := make([]float64, 0, len(solutions))
	for k := range solutions {
		keys = append(keys, k)
	}
	slices.Sort(keys)

	orderedSolutions := make([]basicNode, 0)
	for _, key := range keys {
		orderedSolutions = append(orderedSolutions, basicNode{q: solutions[key], cost: key})
	}
	return orderedSolutions, nil
}

func getTagPoses(
	ctx context.Context,
	a arm.Arm,
	pt posetracker.PoseTracker,
	positions [][]referenceframe.Input,
) ([]referenceframe.FrameSystemPoses, [][]referenceframe.Input, error) {
	allPoses := make([]referenceframe.FrameSystemPoses, 0, len(positions))
	actualPositions := make([][]referenceframe.Input, 0)
	for _, position := range positions {
		err := a.MoveToJointPositions(ctx, position, nil)
		if err != nil {
			return nil, nil, err
		}
		time.Sleep(time.Second)
		j, err := averageJointPosition(ctx, a, 100)
		if err != nil {
			return nil, nil, err
		}
		actualPositions = append(actualPositions, j)

		poses, err := pt.Poses(ctx, nil, nil)
		if err != nil {
			return nil, nil, err
		}
		allPoses = append(allPoses, poses)
	}
	return allPoses, actualPositions, nil
}

func averageJointPosition(ctx context.Context, a arm.Arm, n int) ([]referenceframe.Input, error) {
	avg := make([]referenceframe.Input, len(a.ModelFrame().DoF()))
	for range n {
		j, err := a.JointPositions(ctx, nil)
		if err != nil {
			return nil, err
		}
		for i := range len(avg) {
			avg[i].Value += j[i].Value
		}
	}
	for i := range len(avg) {
		avg[i].Value /= float64(n)
	}
	return avg, nil
}

func printWorldStatePoses(
	a arm.Arm,
	poses []referenceframe.FrameSystemPoses,
	unknownPose spatialmath.Pose,
	tags []string,
	calibrationPositions [][]referenceframe.Input,
) error {
	for _, tag := range tags {
		fmt.Printf("Tag #%s\n", tag)
		for i, pose := range poses {
			tagPose, ok := pose[tag]
			if !ok {
				continue
			}
			fmt.Printf("\tPose #%d\t", i)
			armPose, err := a.ModelFrame().Transform(calibrationPositions[i])
			if err != nil {
				fmt.Printf("\n")
				return err
			}
			worldPose := spatialmath.Compose(spatialmath.Compose(armPose, unknownPose), tagPose.Pose())
			fmt.Printf("%.3v\n", worldPose)
		}
	}
	return nil
}

func saveToFile(filename string, data any) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(data)
	if err != nil {
		return err
	}
	return nil
}

func loadFromFile[T any](filename string, data *T) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&data)
	if err != nil {
		return err
	}
	return nil
}

func posesToProtobuf(poses []referenceframe.FrameSystemPoses) []map[string]commonpb.Pose {
	pb := make([]map[string]commonpb.Pose, 0)
	for _, fsPose := range poses {
		m := make(map[string]commonpb.Pose)
		for key, value := range fsPose {
			m[key] = *spatialmath.PoseToProtobuf(value.Pose())
		}
		pb = append(pb, m)
	}
	return pb
}

func protobufToPoses(pb []map[string]*commonpb.Pose) []referenceframe.FrameSystemPoses {
	poses := make([]referenceframe.FrameSystemPoses, 0)
	for _, p := range pb {
		m := make(referenceframe.FrameSystemPoses)
		for key, value := range p {
			m[key] = referenceframe.NewPoseInFrame("cam", spatialmath.NewPoseFromProtobuf(value))
		}
		poses = append(poses, m)
	}
	return poses
}

func poseToFloats(p spatialmath.Pose) []float64 {
	pt := p.Point()
	o := p.Orientation().AxisAngles().ToR3()
	return []float64{pt.X, pt.Y, pt.Z, o.X, o.Y, o.Z}
}

func floatsToPose(f []float64) spatialmath.Pose {
	return spatialmath.NewPose(r3.Vector{X: f[0], Y: f[1], Z: f[2]}, spatialmath.R3ToR4(r3.Vector{f[3], f[4], f[5]}))
}
